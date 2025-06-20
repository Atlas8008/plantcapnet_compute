

def check_if_script():
    # Check, if the code is run in notebook or in script
    try:
        # This function only exists in a notebook
        get_ipython()

        return False
        # --> We are running in a notebook
    except NameError:
        # --> We are running in a script
        return True