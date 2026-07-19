import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import re
import threading
from json_repair import repair_json
from dotenv import load_dotenv
import sqlite3
import time
import uuid
from typing import Literal

try:
    from langchain_community.llms import MLXPipeline
    from mlx_lm import load, generate
except ImportError:
    MLXPipeline = None  # type: ignore[misc, assignment]
    load = generate = None  # type: ignore[misc, assignment]

try:
    from google import genai
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    genai = None  # type: ignore[misc, assignment]
    ChatGoogleGenerativeAI = None  # type: ignore[misc, assignment]

try:
    import psycopg
    from psycopg.types.json import Jsonb
except ImportError:
    psycopg = None  # type: ignore[misc, assignment]
    Jsonb = None  # type: ignore[misc, assignment]

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None  # type: ignore[misc, assignment]

_SQLITE_LOCK = threading.Lock()
_JUDGE_RETRY_HINT = (
    "\n\nIMPORTANT: Respond with valid JSON only. No markdown fences, no preamble."
)


def resolve_db_backend(database_url: str, db_backend: str | None = None) -> Literal["postgres", "sqlite"]:
    if db_backend in ("postgres", "sqlite"):
        return db_backend  # type: ignore[return-value]
    if database_url.endswith(".db") or database_url.startswith("sqlite:"):
        return "sqlite"
    return "postgres"


class RAGHelper:

    def __init__(self, api_key):
        self.api_key = api_key
        self.search_type = None
        self.search_kwargs = dict()
        self.gen_model = None
        self.eval_model = None
        self.embed_model = None
        self.response = None
        self.sent_list = list()
        self.relevance = 0.0
        self.utilization = 0.0
        self.completeness = 0.0
        self.adherence = None
        self.parse_error = False
        self.eval_message = ["""You will review a response to a question, checking whether it's supported by a set of source documents. Documents are split into sentences, each with a key (e.g. 'a0', 'b0').

Documents:
{documents}

Question:
{question}

Response to evaluate:
{answer}

Return a single JSON object with this schema:
{{
"relevance_explanation": string,
"all_relevant_sentence_keys": [string],
"overall_supported_explanation": string,
"overall_supported": boolean,
"sentence_support_information": [
{{
"response_sentence_key": string,
"explanation": string,
"supporting_sentence_keys": [string],
"fully_supported": boolean
}}
],
"all_utilized_sentence_keys": [string]
}}

Field instructions:

- relevance_explanation: Step-by-step, explain which document sentences are useful for answering the question — based on the question and documents alone, independent of the response.
- all_relevant_sentence_keys: All sentence keys relevant to answering the question (used in the response or not). Exclude sentences that wouldn't affect answerability if removed.
- overall_supported_explanation: Assess each claim in the response individually first, then conclude whether the response as a whole is supported.
- overall_supported: Boolean matching the conclusion above.
- sentence_support_information: One entry per response sentence:
  - response_sentence_key: the sentence's identifier
  - explanation: why it is/isn't supported
  - supporting_sentence_keys: document keys backing the sentence. Empty if unsupported. Use these special values where applicable instead of real keys: "supported_without_sentence" (valid "cannot answer" statements), "general" (transitions/summaries/framing), "well_known_fact" (established facts, e.g. formulas), "numerical_reasoning" (math performed by the model).
  - fully_supported: false if supporting_sentence_keys is empty; otherwise true only if the sentence is entirely backed by the cited text, false if only partially backed.
- all_utilized_sentence_keys: All sentence keys actually used (directly or implicitly) to build the answer. Exclude anything removable without affecting the answer.

Respond with valid JSON only — properly escaped quotes and newlines, no preamble, no postamble, no code fences.
        """]

    def _make_chat_model(self, model: str, model_type: str, temperature: float = 0.0):
        if model_type == "ollama":
            return ChatOllama(model=model, temperature=temperature)
        if model_type == "openai":
            if ChatOpenAI is None:
                raise RuntimeError("langchain-openai is not installed")
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set")
            return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)
        return ChatGroq(model=model, temperature=temperature, api_key=self.api_key)

    def get_embedding_model_name(self, retriever):
        # 1. Get the vector store
        vs = retriever.vectorstore
        
        # 2. Get the embedding object (checking all common attribute names)
        embed_obj = getattr(vs, "embeddings", None) or \
                    getattr(vs, "embedding_func", None) or \
                    getattr(vs, "_embedding_function", None)
                    
        if not embed_obj:
            return "Unknown_Embeddings"
            
        # 3. Get the model string (checking common provider attribute names)
        model_name = getattr(embed_obj, "model_name", None) or \
                     getattr(embed_obj, "model", None) or \
                     "Custom_or_Unknown_Model"
        return model_name



    def retriever_fn(self, retriever, query: str):
        relevant_docs = retriever.invoke(query)
        context = "\n".join([d.page_content for d in relevant_docs])
        # print(relevant_docs)
        sent_list = list()
        for i, d in enumerate(relevant_docs):
            index_char = chr(97+i)
            for j, s in enumerate(d.page_content.split(".")):
                sent_list.append([index_char+str(j), s])
        self.search_type = retriever.search_type
        self.search_kwargs = retriever.search_kwargs
        self.embed_model = self.get_embedding_model_name(retriever)
     

        return (context, sent_list)
        
        

    def simple_rag(self, query: str, expert_domain: str, retriever,
                   gen_model: str="llama-3.1-8b-instant", temperature: float=0.0,
                   model_type: str = "groq"):
        messages = [
        ("system", """You are a {expert_domain} expert. You are given a question and a list of documents and need to
        answer the question. Answer the question only based on these documents. These
        documents can help you answer the question: {context}. If you are not sure about the
        answer, you can say 'I don't know' or 'I don't know the answer to that question.'"""),
        ("human", "{question}"),
        ]

        q_prompt = ChatPromptTemplate(messages=messages)
        q_model = self._make_chat_model(gen_model, model_type, temperature)
        q_chain = q_prompt | q_model | StrOutputParser()

        context, sent_list = self.retriever_fn(retriever=retriever, query=query)
    
        response = q_chain.invoke({"question": query, "context": context, "expert_domain": expert_domain})
        self.gen_model = gen_model
        self.response = response
        self.sent_list = sent_list

        return (query, response, sent_list)


    def evaluate_rag(self, query: str, response: str, sent_list: list, eval_model: str,
                     temperature: float=0.0, model_type: str = "groq"):
        messages = self.eval_message

        eval_prompt = ChatPromptTemplate(messages=messages)
        e_model = self._make_chat_model(eval_model, model_type, temperature)
        eval_chain = eval_prompt | e_model | StrOutputParser()

        # context, sent_list = self.retriever_fn(retriever=retriever, query=query)
    
        eval_response = eval_chain.invoke({"documents":sent_list, "question":query, "answer":response})
        self.eval_model = eval_model

        return eval_response

    def evaluate_and_score_with_retry(
        self,
        query: str,
        response: str,
        sent_list: list,
        eval_model: str,
        *,
        model_type: str = "groq",
        max_retries: int = 3,
    ) -> tuple[str, float | None, float | None, float | None, bool | None]:
        """Run judge LLM with retries when JSON parsing fails."""
        self.parse_error = False
        last_eval = ""
        for attempt in range(1, max_retries + 1):
            try:
                last_eval = self.evaluate_rag(
                    query=query,
                    response=response,
                    sent_list=sent_list,
                    eval_model=eval_model,
                    model_type=model_type,
                )
                relevance, utilization, completeness, adherence = self.get_metrics(
                    last_eval, sent_list
                )
                return last_eval, relevance, utilization, completeness, adherence
            except Exception as err:
                print(f"Judge parse attempt {attempt}/{max_retries} failed: {err}")
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                    response_for_judge = response + _JUDGE_RETRY_HINT
                    last_eval = self.evaluate_rag(
                        query=query,
                        response=response_for_judge,
                        sent_list=sent_list,
                        eval_model=eval_model,
                        model_type=model_type,
                    )
                    try:
                        relevance, utilization, completeness, adherence = self.get_metrics(
                            last_eval, sent_list
                        )
                        return last_eval, relevance, utilization, completeness, adherence
                    except Exception:
                        continue
        self.parse_error = True
        self.relevance = None
        self.utilization = None
        self.completeness = None
        self.adherence = None
        return last_eval, None, None, None, None

    def db_insert(self,
              chunk_size: int,
              chunk_overlap: int,
              table_name: str, database_url: str,
              qid: int, vector_db: str, session_id,
                  retrieval_type: str,
                  db_backend: str | None = None
              
       
             ):
        backend = resolve_db_backend(database_url, db_backend)
        session_id = str(session_id)
        context = [str(sent) for sent in self.sent_list] if isinstance(self.sent_list, list) else [str(self.sent_list)]
        adherence_val = int(self.adherence) if self.adherence is not None else None
        parse_error = int(bool(self.parse_error))

        columns = """
                  session_id, id, chunk_size, chunk_overlap, vector_db,
                  retreival_type, gen_model, embed_model, eval_model,
                  search_type, search_kwargs, response, context,
                  adherence, relevance, utilization, completeness, parse_error
        """
        values_common = (
            session_id,
            qid,
            chunk_size,
            chunk_overlap,
            vector_db,
            retrieval_type,
            self.gen_model,
            self.embed_model,
            self.eval_model,
            self.search_type,
        )

        try:
            if backend == "sqlite":
                sqlite_path = database_url.removeprefix("sqlite:")
                conn = sqlite3.connect(sqlite_path)
                cursor = conn.cursor()
                sql_query = f"""
                    INSERT INTO {table_name} ({columns})
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
                row = values_common + (
                    json.dumps(self.search_kwargs),
                    self.response,
                    json.dumps(context),
                    adherence_val,
                    self.relevance,
                    self.utilization,
                    self.completeness,
                    parse_error,
                )
                with _SQLITE_LOCK:
                    cursor.execute(sql_query, row)
                    conn.commit()
                conn.close()
                return

            if psycopg is None:
                raise RuntimeError("psycopg is not installed")
            conn = psycopg.connect(database_url)
            cursor = conn.cursor()
            sql_query = f"""
                INSERT INTO {table_name} (
                  session_id, id, chunk_size, chunk_overlap, vector_db,
                  retreival_type, gen_model, embed_model, eval_model,
                  search_type, search_kwargs, response, context,
                  adherence, relevance, utilization, completeness
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            row = values_common + (
                Jsonb(self.search_kwargs),
                self.response,
                context,
                adherence_val,
                self.relevance,
                self.utilization,
                self.completeness,
            )
            cursor.execute(sql_query, row)
            conn.commit()
            conn.close()

        except Exception as e:
            print(e)
            if "conn" in locals():
                conn.close()

    def get_metrics(self, e, sl):
        relevance = 0.0
        utilization = 0.0
        adherence = 0.0
        completeness = 0.0
        # print(e)
        # e = json.loads(e)
        # print(type(e))
        # print(e)
        # print("+"*50)
        match = re.search(r'\{.*\}', e, re.DOTALL)
        if match:
            e_raw = match.group(0)
        else:
            e_raw = e
        # try:
        data = json.loads(repair_json(e_raw))
        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                e = data[0]
            else:
                raise ValueError("Received a list with no dictionary inside.")
        else:
            e = data
    
        total_sentences_len = len(sl)
        len_all_relevant_sentence_keys = len(e["all_relevant_sentence_keys"])
        len_all_utilized_sentence_keys = len(e["all_utilized_sentence_keys"])
        relevant_set = set(e["all_relevant_sentence_keys"])
        utilized_set = set(e["all_utilized_sentence_keys"])
        intersect_keys = relevant_set.intersection(utilized_set)
        len_intersect_keys = len(intersect_keys)
        sentence_support_information = [s.get("fully_supported") for s in e["sentence_support_information"]]
        
        relevance = len_all_relevant_sentence_keys/total_sentences_len if total_sentences_len > 0 else 0.0
        utilization = len_all_utilized_sentence_keys/total_sentences_len if total_sentences_len > 0 else 0.0
        completeness = len_intersect_keys/len_all_relevant_sentence_keys if len_all_relevant_sentence_keys > 0 else 0.0
        adherence = all(sentence_support_information)
        self.relevance = relevance
        self.utilization = utilization
        self.completeness = completeness
        self.adherence = adherence
    
        return relevance, utilization, completeness, adherence

    def get_output(self, query: str, eval_message: list, retriever, expert_domain: str = "finance", gen_model="llama-3.1-8b-instant", evaluate_model="llama-3.3-70b-versatile",
                   model_type="ollama"):
        # print("got into output")
        output = self.simple_rag(query=query, eval_message=eval_message, expert_domain=expert_domain, retriever=retriever,
                            gen_model=gen_model, evaluate_model=evaluate_model, model_type=model_type)
        r, e, sl = output
        # print("output retrieved")
        relevance, utilization, completeness, adherence = self.get_metrics(e, sl, r)
        # print(f"""Question: {query}\nAnswer: {r}\n\n{'='*50}\n\nScores: \nAdherence: {adherence}\nRelevance: {relevance:.2f}\nUtilization: {utilization:.2f}\nCompleteness: {completeness:.2f}
        # \n\n{'='*50}\n\noutput_evalution: \n{e}""")
        return (relevance, utilization, completeness, adherence)

    def calculate_metrics(self, query_list: list, eval_message: list, retriever, expert_domain: str = "finance", gen_model="llama-3.1-8b-instant", evaluate_model="llama-3.3-70b-versatile", 
                     model_type="ollama"):
        adherence_score = list()
        relevance_score = list()
        utilization_score = list()
        completeness_score = list()
        num_queries = len(query_list)
        print(f"Total no. of queries: {num_queries}")
    
        for i, q in enumerate(query_list, start=1):
            r, u, c, a = self.get_output(query=q, eval_message=eval_message, expert_domain=expert_domain, retriever=retriever, evaluate_model=evaluate_model, model_type=model_type)
            relevance_score.append(r)
            utilization_score.append(u)
            completeness_score.append(c)
            adherence_score.append(a)
            print(f"{i}/{num_queries} completed")
    
        rag_adherence_score_mean = np.mean(adherence_score)
        rag_relevance_score_mean = np.mean(relevance_score)
        rag_utilization_score_mean = np.mean(utilization_score)
        rag_completeness_score_mean = np.mean(completeness_score)
        print(f"rag_adherence_score_mean: {rag_adherence_score_mean:.2f}")
        print(f"rag_relevance_score_mean: {rag_relevance_score_mean:.2f}")
        print(f"rag_utilization_score_mean: {rag_utilization_score_mean:.2f}")
        print(f"rag_completeness_score_mean: {rag_completeness_score_mean:.2f}")


    # def db_insert(self, dataset, eval_message, retriever,
    #           chunk_size=1024, dataset_name="finqa",
    #           vector_db="chroma", embed_model="BAAI/bge-base-en-v1.5",
    #           chunk_overlap=200, gen_model="llama-3.1-8b-instant",
    #           eval_model="llama-3.3-70b-versatile",
    #           eval_model_type="groq",
    #           table_name="rag_table", sample=100,
    #           sqldb="../sqldb/ragproject.db"
                
              
              
              
    #          ):
    #     try:
    #         conn = sqlite3.connect(sqldb)
    #         cursor = conn.cursor()
    #         quest_dict = dict()
    #         j = 1
            
    #         for d in dataset:
    #             # print(d)
    #             # print(quest_dict)
    #             for ds in dataset:
    #                 if ds["question"] == d["question"]:
    #                     if ds["question"] in quest_dict:
    #                         if ds["generation_model_name"].startswith("gpt"):
    #                             quest_dict[ds["question"]].update({
    #                                 "gpt_model_adherence": d["adherence_score"],
    #                                 "gpt_model_relevance_score": d["relevance_score"],
    #                                 "gpt_model_utilization_score": d["utilization_score"],
    #                                 "gpt_model_completeness_score": d["completeness_score"]
    #                             })
    #                         elif ds["generation_model_name"].startswith("claude"):
    #                             quest_dict[ds["question"]].update({
    #                                 "claude_model_adherence": ds["adherence_score"],
    #                                 "claude_model_relevance_score": ds["relevance_score"],
    #                                 "claude_model_utilization_score": ds["utilization_score"],
    #                                 "claude_model_completeness_score": ds["completeness_score"]
    #                             })
    #                     else:
    #                         if ds["generation_model_name"].startswith("gpt"):
    #                             quest_dict[ds["question"]] = {
    #                                 "gpt_model_adherence": ds["adherence_score"],
    #                                 "gpt_model_relevance_score": ds["relevance_score"],
    #                                 "gpt_model_utilization_score": ds["utilization_score"],
    #                                 "gpt_model_completeness_score": ds["completeness_score"]
    #                             }
    #                         elif ds["generation_model_name"].startswith("claude"):
    #                             quest_dict[ds["question"]] = {
    #                                 "claude_model_adherence": ds["adherence_score"],
    #                                 "claude_model_relevance_score": ds["relevance_score"],
    #                                 "claude_model_utilization_score": ds["utilization_score"],
    #                                 "claude_model_completeness_score": ds["completeness_score"]
    #                     }
    #             if j == sample:
    #                 break
    #             else:
    #                 j += 1
        
    #         # return quest_dict
    #         questions_count = len(quest_dict.keys())
    #         i = 1
    #         for q, v in quest_dict.items():
    #             metadata = f"""
    #             {{
    #             chunk_size: {chunk_size},
    #             dataset_name: {dataset_name},
    #             vector_db: {vector_db},
    #             embed_model: {embed_model},
    #             chunk_overlap: {chunk_overlap},
    #             gen_model: {gen_model},
    #             eval_model: {eval_model},
    #             retriever_search_type: {retriever.search_type},
    #             retriever_search_args: {retriever.search_kwargs}
    #             }}
    #             """
    #             question = q
    #             gpt_model_adherence = v.get("gpt_model_adherence", 0)
    #             gpt_model_relevance_score = v.get("gpt_model_relevance_score", 0)
    #             gpt_model_utilization_score = v.get("gpt_model_utilization_score", 0)
    #             gpt_model_completeness_score = v.get("gpt_model_completeness_score", 0)
    #             claude_model_adherence = v.get("claude_model_adherence", 0)
    #             claude_model_relevance_score = v.get("claude_model_relevance_score", 0)
    #             claude_model_utilization_score = v.get("claude_model_utilization_score", 0)
    #             claude_model_completeness_score = v.get("claude_model_completeness_score", 0)
        
    #             relevance, utilization, completeness, adherence = self.get_output(query=question, eval_message=eval_message,
    #                        retriever=retriever, evaluate_model=eval_model,
    #                        model_type=eval_model_type, gen_model=gen_model)
                
        
    #             sql_query = f"""
    #                 INSERT INTO {table_name} (
    #                     metadata, question, gpt_model_adherence, gpt_model_relevance_score, 
    #                     gpt_model_utilization_score, gpt_model_completeness_score, claude_model_adherence, 
    #                     claude_model_relevance_score, claude_model_utilization_score, 
    #                     claude_model_completeness_score, adherence, relevance_score, 
    #                     utilization_score, completeness_score
    #                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    #             """
        
    #             values = (
    #             str(metadata),  # Ensure these are strings or appropriate types
    #             str(question), 
    #             gpt_model_adherence, 
    #             gpt_model_relevance_score, 
    #             gpt_model_utilization_score, 
    #             gpt_model_completeness_score, 
    #             claude_model_adherence, 
    #             claude_model_relevance_score, 
    #             claude_model_utilization_score, 
    #             claude_model_completeness_score, 
    #             adherence, 
    #             relevance, 
    #             utilization, 
    #             completeness
    #             )
    #             cursor.execute(sql_query, values)
    #             conn.commit()
    #             print(f"{i}/{questions_count} completed")
    #             i += 1
    #             time.sleep(10)
        
    #         conn.close()
        
    #         print(f"All entries inserted")

    #     except Exception as e:
    #         print(e)
    #         conn.close()




    def get_scores(self, queries: list, eval_message, retriever,no_of_samples=1, evaluate_model="qwen2.5:7b", model_type="ollama"):
        qsubset = random.sample(queries, no_of_samples)
        # print(qsubset)
        adherence_score = list()
        relevance_score = list()
        utilization_score = list()
        completeness_score = list()
        for i, q in enumerate(ragbench_finqa["question"]):
            if q in qsubset and ragbench_finqa[i]['generation_model_name']=="gpt-3.5-turbo-0125":
                # print(q)
                # print(ragbench_finqa[i]['generation_model_name'])
                adherence_score.append(int(ragbench_finqa[i]["adherence_score"]))
                relevance_score.append(ragbench_finqa[i]["relevance_score"])
                utilization_score.append(ragbench_finqa[i]["utilization_score"])
                completeness_score.append(ragbench_finqa[i]["completeness_score"])
        adherence_score_mean = np.mean(adherence_score)
        relevance_score_mean = np.mean(relevance_score)
        utilization_score_mean = np.mean(utilization_score)
        completeness_score_mean = np.mean(completeness_score)
        print(f"adherence_score_mean: {adherence_score_mean:.2f}")
        print(f"relevance_score_mean: {relevance_score_mean:.2f}")
        print(f"utilization_score_mean: {utilization_score_mean:.2f}")
        print(f"completeness_score_mean: {completeness_score_mean:.2f}")
        print("="*100)
    
        self.calculate_metrics(query_list=qsubset, eval_message=eval_message, retriever=retriever, evaluate_model=evaluate_model, model_type=model_type)




