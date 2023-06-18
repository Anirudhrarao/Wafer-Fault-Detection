import os
import sys


def error_message_details(error,error_detail:sys):
    '''
        Desc: This method will catch our error with location
    '''
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error ocurred in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
    )

    return error_message
class CustomException(Exception):
    '''
        Desc:This is CustomException class which is inherited with parent class Exception
            basically here we called our init function of parent class and passed error 
            message in Exception class
    '''

    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self) -> str:
        return self.error_message
    
