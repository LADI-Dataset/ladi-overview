import random
import string

def get_random_string(length):
    '''
    generates a random string of given length, includes letters and digits
    '''
    letters = string.ascii_letters + string.digits
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str