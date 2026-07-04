
import traceback

class ExceptionHelper:
    @staticmethod
    def get_exception_message(e: Exception) -> str:
        """
        Get a formatted exception message with the exception type.
        
        Args:
            e: The exception object
            
        Returns:
            Formatted string with exception message and type
        """

        
        ret_msg = f"❌ Exception:\n{str(e)}\n{type(e).__name__}"
        return ret_msg
    
    @staticmethod
    def get_exception_traceback(e: Exception) -> str:
        """
        Get the full traceback of an exception as a string.
        
        Args:
            e: The exception object
            
        Returns:
            Formatted traceback string
        """
        ret_traceback = traceback.format_exc()
        return ret_traceback
