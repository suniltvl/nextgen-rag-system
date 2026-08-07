"""One-off generator for the RAG dashboard's sample dataset.

Produces sample_data/domains.json and sample_data/questions.json with
5 domains x 10 questions each, mimicking the shape a real RAGBench-backed
pipeline would produce (answer, ground truth, retrieved chunks, pipeline
config, evaluation metrics, latency). Re-run this script any time you want
a fresh randomized dataset:

    python gradio_app/generate_sample_data.py
"""

from __future__ import annotations

import json
import random
import textwrap
import uuid
from pathlib import Path

random.seed(42)

SAMPLE_DATA_DIR = Path(__file__).parent / "sample_data"

DOMAINS = [
    {"id": "biomedical", "name": "Biomedical Research"},
    {"id": "customer_support", "name": "Customer Support"},
    {"id": "finance", "name": "Finance"},
    {"id": "general_knowledge", "name": "General Knowledge"},
    {"id": "legal", "name": "Legal"},
]

QUESTIONS = {
    "biomedical": [
        ("What is CRISPR?", "CRISPR is a gene-editing technology that allows scientists to precisely alter DNA sequences by using a guide RNA and the Cas9 enzyme to cut and modify targeted genes."),
        ("Explain mRNA vaccines.", "mRNA vaccines deliver a synthetic strand of messenger RNA that instructs cells to produce a harmless viral protein, training the immune system to recognize and fight the real pathogen."),
        ("What is apoptosis?", "Apoptosis is programmed cell death, a controlled process the body uses to eliminate damaged or unneeded cells without triggering inflammation."),
        ("What is PCR?", "PCR, or polymerase chain reaction, is a laboratory technique used to amplify small segments of DNA into millions of copies for analysis."),
        ("Difference between DNA and RNA.", "DNA is a double-stranded molecule that stores genetic information long-term, while RNA is typically single-stranded and carries out instructions for protein synthesis."),
        ("What is Gene Therapy?", "Gene therapy involves introducing, removing, or altering genetic material within a patient's cells to treat or prevent disease."),
        ("What is Protein Folding?", "Protein folding is the physical process by which a protein chain acquires its functional three-dimensional structure, determined largely by its amino acid sequence."),
        ("Explain Genome Sequencing.", "Genome sequencing determines the complete DNA sequence of an organism's genome at a single time, revealing the order of all its nucleotide bases."),
        ("What is Immunotherapy?", "Immunotherapy is a cancer treatment that helps a patient's immune system recognize and attack cancer cells more effectively."),
        ("What is Stem Cell Therapy?", "Stem cell therapy uses undifferentiated cells capable of developing into various cell types to repair or replace damaged tissue."),
    ],
    "customer_support": [
        ("How do I reset my password?", "You can reset your password by clicking 'Forgot Password' on the login page and following the email verification link sent to your registered address."),
        ("How do I cancel my subscription?", "You can cancel your subscription anytime from Account Settings > Billing > Cancel Plan; the cancellation takes effect at the end of the current billing cycle."),
        ("How can I update my billing address?", "Billing addresses can be updated under Account Settings > Payment Methods by editing the address associated with your saved card."),
        ("Refund Policy", "Our refund policy allows a full refund within 30 days of purchase for unused services, processed back to the original payment method within 5-7 business days."),
        ("Shipping Delay", "Shipping delays are typically caused by high order volume or carrier disruptions; you can track real-time status using the tracking link in your confirmation email."),
        ("Track my order", "You can track your order by visiting the Orders section of your account and clicking 'Track Package' next to the relevant order."),
        ("Change email address", "To change your email address, go to Account Settings > Profile and enter your new email; a verification link will be sent to confirm the change."),
        ("Update phone number", "Your phone number can be updated in Account Settings > Profile > Contact Information at any time."),
        ("Download invoice", "Invoices are available for download under Account Settings > Billing > Invoice History in PDF format."),
        ("Upgrade my plan", "You can upgrade your plan instantly from Account Settings > Billing > Change Plan, with prorated charges applied to your next invoice."),
    ],
    "finance": [
        ("What is EBITDA?", "EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization, used to evaluate a company's core operating profitability."),
        ("What is CAPEX?", "CAPEX, or capital expenditure, refers to funds a company uses to acquire, upgrade, or maintain physical assets like property or equipment."),
        ("Explain NPV.", "Net Present Value (NPV) measures the difference between the present value of cash inflows and outflows over time, used to assess investment profitability."),
        ("Difference between Equity and Debt.", "Equity represents ownership in a company with no repayment obligation, while debt is borrowed capital that must be repaid with interest."),
        ("What is Operating Margin?", "Operating margin measures the percentage of revenue remaining after paying variable production costs, indicating operational efficiency."),
        ("What is Free Cash Flow?", "Free cash flow is the cash a company generates after accounting for capital expenditures, available for distribution to investors or reinvestment."),
        ("What is ROE?", "Return on Equity (ROE) measures a company's profitability relative to shareholders' equity, showing how efficiently equity capital is used."),
        ("Explain Working Capital.", "Working capital is the difference between a company's current assets and current liabilities, reflecting short-term financial health."),
        ("What is Inflation?", "Inflation is the rate at which the general price level of goods and services rises, eroding purchasing power over time."),
        ("What is Dividend Yield?", "Dividend yield is a financial ratio showing how much a company pays in dividends relative to its share price."),
    ],
    "general_knowledge": [
        ("Capital of Australia?", "The capital of Australia is Canberra, a planned city chosen as a compromise between rivals Sydney and Melbourne."),
        ("Who invented the telephone?", "Alexander Graham Bell is credited with inventing the first practical telephone and was awarded the first US patent for the device in 1876."),
        ("Explain Quantum Computing.", "Quantum computing uses quantum bits, or qubits, which can represent multiple states simultaneously, enabling certain computations to be performed far faster than classical computers."),
        ("Explain Photosynthesis.", "Photosynthesis is the process by which plants convert light energy, water, and carbon dioxide into glucose and oxygen."),
        ("What is Machine Learning?", "Machine learning is a subset of artificial intelligence where systems learn patterns from data to make predictions or decisions without being explicitly programmed."),
        ("Who discovered Penicillin?", "Alexander Fleming discovered penicillin in 1928 when he noticed mold accidentally contaminating a bacterial culture had killed the surrounding bacteria."),
        ("Largest Ocean?", "The Pacific Ocean is the largest and deepest of Earth's oceans, covering more area than all of Earth's land combined."),
        ("Speed of Light?", "The speed of light in a vacuum is approximately 299,792 kilometers per second, a fundamental constant of physics."),
        ("What is Blockchain?", "Blockchain is a distributed ledger technology that records transactions across many computers so that records cannot be altered retroactively."),
        ("Explain Artificial Intelligence.", "Artificial intelligence refers to computer systems designed to perform tasks that normally require human intelligence, such as reasoning, perception, and language understanding."),
    ],
    "legal": [
        ("What is Contract Law?", "Contract law governs the creation and enforcement of agreements between parties, ensuring promises made are legally binding and enforceable."),
        ("Difference between Civil and Criminal Law.", "Civil law resolves disputes between private parties seeking compensation, while criminal law addresses offenses against the state punishable by fines or imprisonment."),
        ("What is Arbitration?", "Arbitration is a form of alternative dispute resolution where an impartial arbitrator reviews evidence and renders a binding decision outside of court."),
        ("What is Tort?", "A tort is a civil wrong that causes harm or loss to another person, giving rise to legal liability for the person who commits it."),
        ("What is Negligence?", "Negligence is a failure to exercise the level of care that a reasonably prudent person would under similar circumstances, resulting in harm to another."),
        ("What is Intellectual Property?", "Intellectual property refers to legal rights protecting creations of the mind, including inventions, literary works, designs, and trademarks."),
        ("What is Copyright?", "Copyright grants creators exclusive rights to reproduce, distribute, and display their original works for a specified period of time."),
        ("What is Trademark?", "A trademark is a recognizable sign, design, or expression that identifies products or services from a particular source and distinguishes them from others."),
        ("What is Consideration?", "Consideration is something of value exchanged between parties in a contract, a required element for the contract to be legally enforceable."),
        ("What is Breach of Contract?", "A breach of contract occurs when one party fails to fulfill their obligations under the terms of a legally binding agreement."),
    ],
}

DOC_NAME_BANK = {
    "biomedical": ["Cell_Biology_Handbook.pdf", "Journal_of_Molecular_Medicine.pdf", "Genomics_Review_2024.pdf", "Immunology_Textbook.pdf", "Clinical_Trials_Digest.pdf", "NIH_Research_Brief.pdf"],
    "customer_support": ["Support_Knowledge_Base.pdf", "Billing_FAQ.pdf", "Account_Policies.pdf", "Shipping_Guide.pdf", "Terms_of_Service.pdf", "Help_Center_Articles.pdf"],
    "finance": ["Annual_Report_2024.pdf", "Investment_Fundamentals.pdf", "Corporate_Finance_Textbook.pdf", "Market_Analysis_Digest.pdf", "SEC_Filing_10K.pdf", "Financial_Ratios_Guide.pdf"],
    "general_knowledge": ["Encyclopedia_Britannica.pdf", "World_Almanac_2024.pdf", "Science_Digest.pdf", "History_Reference.pdf", "Wikipedia_Export.pdf", "National_Geographic_Facts.pdf"],
    "legal": ["Contract_Law_Casebook.pdf", "Civil_Procedure_Guide.pdf", "Legal_Dictionary.pdf", "Supreme_Court_Digest.pdf", "Torts_Handbook.pdf", "IP_Law_Review.pdf"],
}

FILLER_SENTENCES = {
    "biomedical": [
        "Recent advances in molecular biology have enabled researchers to study cellular mechanisms with unprecedented precision.",
        "Laboratory protocols typically require strict temperature and pH control to preserve sample integrity during analysis.",
        "Clinical studies suggest that early diagnosis significantly improves patient outcomes across a wide range of conditions.",
        "The underlying biological pathway involves a cascade of enzymatic reactions regulated by feedback inhibition.",
        "Peer-reviewed literature emphasizes the importance of reproducibility when validating experimental results.",
        "Researchers continue to explore how genetic variation contributes to individual differences in disease susceptibility.",
        "Modern sequencing platforms have dramatically reduced the cost and time required to analyze genetic material.",
        "Ethical review boards oversee human subject research to ensure participant safety and informed consent.",
    ],
    "customer_support": [
        "Our support team aims to resolve most account issues within one business day of the initial request.",
        "Customers are encouraged to check the help center before submitting a new support ticket.",
        "Account changes are logged for security purposes and may require identity verification.",
        "Refunds and credits are typically reflected in the customer's account within a few business days.",
        "Escalated tickets are reviewed by a senior support specialist within 24 hours.",
        "Notification preferences can be adjusted at any time from the account dashboard.",
        "Service disruptions are communicated proactively through the status page and email alerts.",
        "Multi-factor authentication is recommended to further secure customer accounts.",
    ],
    "finance": [
        "Quarterly earnings reports provide investors with insight into a company's revenue trends and cost structure.",
        "Analysts often compare valuation multiples across peer companies to assess relative pricing.",
        "Diversification across asset classes is a common strategy for managing portfolio risk.",
        "Interest rate changes set by central banks influence borrowing costs across the economy.",
        "Audited financial statements are required for publicly traded companies under securities regulations.",
        "Cash flow statements reveal how a company generates and spends money across operating, investing, and financing activities.",
        "Credit ratings agencies assess the likelihood that a borrower will default on outstanding debt.",
        "Market volatility can be influenced by macroeconomic indicators such as employment and inflation data.",
    ],
    "general_knowledge": [
        "Historical records indicate that major scientific breakthroughs often build incrementally on earlier discoveries.",
        "Geographic features such as mountain ranges and ocean currents shape regional climate patterns.",
        "Technological innovation has consistently reshaped how societies communicate and exchange information.",
        "Cultural traditions vary widely across regions but often share common underlying human values.",
        "Reference materials are regularly updated to reflect new research findings and historical reassessments.",
        "Educational curricula typically introduce foundational concepts before progressing to specialized topics.",
        "Environmental factors play a significant role in shaping the distribution of species across ecosystems.",
        "Public datasets allow researchers to verify claims and reproduce prior analyses.",
    ],
    "legal": [
        "Court precedents play a central role in shaping how statutes are interpreted in future cases.",
        "Legal practitioners must balance client advocacy with their duties as officers of the court.",
        "Statutory law is enacted by legislatures, while common law develops through judicial decisions over time.",
        "Due process protections require that individuals receive fair notice and an opportunity to be heard.",
        "Jurisdiction determines which court has the authority to hear and decide a particular case.",
        "Legal remedies may include monetary damages, injunctive relief, or specific performance.",
        "Regulatory compliance requires organizations to stay current with evolving legal standards.",
        "Appellate courts review lower court decisions primarily for legal errors rather than factual disputes.",
    ],
}

CHUNK_STRATEGIES = ["Recursive Character Text Splitter", "Sentence-Aware Splitter", "Fixed-Size Token Splitter"]
CHUNK_SIZES = [256, 512, 1024]
CHUNK_OVERLAPS = {256: 50, 512: 100, 1024: 200}
EMBEDDING_MODELS = [("BAAI/bge-large-en-v1.5", 1024), ("BAAI/LLM-Embedder", 768)]
VECTOR_DBS = ["ChromaDB", "Milvus"]
GEN_MODELS = ["openai/gpt-4o-mini", "openai/gpt-5-nano", "qwen/qwen3.7-flash", "meta-llama/llama-3.1-8b-instruct"]
EVAL_MODELS = ["openai/gpt-oss-20b", "meta-llama/llama-3.1-70b-instruct"]


def make_chunk_text(domain_id: str, seed_sentence: str, target_words: tuple[int, int] = (100, 200)) -> str:
    pool = FILLER_SENTENCES[domain_id]
    sentences = [seed_sentence]
    word_count = len(seed_sentence.split())
    low, high = target_words
    shuffled = pool[:]
    random.shuffle(shuffled)
    for sentence in shuffled:
        if word_count >= low:
            break
        sentences.append(sentence)
        word_count += len(sentence.split())
    text = " ".join(sentences)
    words = text.split()
    if len(words) > high:
        text = " ".join(words[:high])
    return textwrap.fill(text, width=100)


def build_pipeline_config() -> dict:
    chunk_size = random.choice(CHUNK_SIZES)
    embedding_model, embedding_dim = random.choice(EMBEDDING_MODELS)
    return {
        "chunk_strategy": random.choice(CHUNK_STRATEGIES),
        "chunk_size": chunk_size,
        "chunk_overlap": CHUNK_OVERLAPS[chunk_size],
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dim,
        "vector_database": random.choice(VECTOR_DBS),
        "retrieval_type": "dense",
        "top_k": 4,
        "generator_llm": random.choice(GEN_MODELS),
        "evaluation_llm": random.choice(EVAL_MODELS),
    }


def build_evaluation_metrics() -> dict:
    faithfulness = round(random.uniform(0.92, 0.99), 3)
    context_relevance = round(random.uniform(0.90, 0.99), 3)
    context_utilization = round(random.uniform(0.88, 0.98), 3)
    answer_completeness = round(random.uniform(0.90, 0.99), 3)
    overall = round(
        0.30 * faithfulness
        + 0.30 * context_relevance
        + 0.20 * context_utilization
        + 0.20 * answer_completeness,
        3,
    )
    return {
        "overall_score": overall,
        "faithfulness": faithfulness,
        "context_relevance": context_relevance,
        "context_utilization": context_utilization,
        "answer_completeness": answer_completeness,
    }


def build_latency() -> dict:
    retrieval_ms = random.randint(40, 220)
    generation_ms = random.randint(400, 2200)
    evaluation_ms = random.randint(300, 1500)
    prompt_tokens = random.randint(350, 1200)
    completion_tokens = random.randint(60, 400)
    return {
        "retrieval_latency_ms": retrieval_ms,
        "generation_latency_ms": generation_ms,
        "evaluation_latency_ms": evaluation_ms,
        "total_response_time_ms": retrieval_ms + generation_ms + evaluation_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def build_retrieved_documents(domain_id: str) -> list[dict]:
    doc_names = random.sample(DOC_NAME_BANK[domain_id], k=4)
    scores = sorted((round(random.uniform(0.82, 0.99), 3) for _ in range(4)), reverse=True)
    seeds = random.sample(FILLER_SENTENCES[domain_id], k=4)
    docs = []
    for rank, (doc_name, score, seed) in enumerate(zip(doc_names, scores, seeds), start=1):
        text = make_chunk_text(domain_id, seed)
        docs.append(
            {
                "rank": rank,
                "similarity_score": score,
                "document_name": doc_name,
                "page_number": random.randint(1, 48),
                "chunk_id": f"{doc_name.split('.')[0].lower()}_chunk_{random.randint(1, 999):03d}",
                "chunk_length": len(text),
                "retrieved_text": text,
            }
        )
    return docs


def main() -> None:
    SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    domains_out = []
    questions_out = []

    for domain in DOMAINS:
        domain_id = domain["id"]
        qa_pairs = QUESTIONS[domain_id]
        domains_out.append({**domain, "question_count": len(qa_pairs)})

        for question_text, ground_truth in qa_pairs:
            questions_out.append(
                {
                    "id": str(uuid.uuid4()),
                    "domain_id": domain_id,
                    "domain_name": domain["name"],
                    "question": question_text,
                    "answer": ground_truth,
                    "ground_truth": ground_truth,
                    "retrieved_documents": build_retrieved_documents(domain_id),
                    "pipeline_config": build_pipeline_config(),
                    "evaluation_metrics": build_evaluation_metrics(),
                    "latency": build_latency(),
                }
            )

    (SAMPLE_DATA_DIR / "domains.json").write_text(json.dumps(domains_out, indent=2), encoding="utf-8")
    (SAMPLE_DATA_DIR / "questions.json").write_text(json.dumps(questions_out, indent=2), encoding="utf-8")

    print(f"Wrote {len(domains_out)} domains and {len(questions_out)} questions to {SAMPLE_DATA_DIR}")


if __name__ == "__main__":
    main()
