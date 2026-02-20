def get_error_info(error, error_detail):
    """
    Used to extract error information from exceptions.
    
    params error: Exception object
    params error_detail: sys module to get exception details

    return: Formatted error message string
    """
    _,_, exctb = error_detail.exc_info()
    filename = exctb.tb_frame.f_code.co_filename
    error_message = 'Error occured in python script file {0}, line number {1}, error message {2}'.format(filename, exctb.tb_lineno, str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = get_error_info(error_message, error_detail)
    
    def __str__(self):
        return self.error_message