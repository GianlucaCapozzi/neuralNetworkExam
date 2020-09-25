#!/usr/bin/env python3
import sys
import subprocess
import configparser


def parse_config():
    config = configparser.ConfigParser()
    config.read('configs.ini')
    return config
def main():
    print("Starting experiments")
    config = parse_config()
    keysFree= [c for c in config["FREE"] ]
    keysFast= [c for c in config["FAST"] ]
    
    freeCommand = ["python3 free.py"] + ["--"+k+" "+config["FREE"][k] for k in keysFree]
    fastCommand = ["python3 fast.py"] + ["--"+k+" "+config["FAST"][k] for k in keysFast]
    print("-"*40,"FREE","-"*40)
    
    for k in keysFree:
        print(k,config["FREE"][k])
    subprocess.call(freeCommand, shell=True)

    print("-"*40,"FAST","-"*40)
    for k in keysFast:
        print(k,config["FAST"][k])
    subprocess.call(fastCommand, shell=True)



def __main__(function, path, user_args):
    """Wraps a main function to display a usage message when necessary."""
    
    co = function.__code__
    
    num_args = co.co_argcount

    if function.__defaults__ is not None:
        min_args = num_args - len(function.__defaults__)
    else:
        min_args = num_args
    
    if co.co_flags & 0x04: # function captures extra arguments with *
        max_args = None
    else:
        max_args = num_args
    
    if min_args <= len(user_args) and (max_args is None or
                                       max_args >= len(user_args)):
        return(function(*user_args))

    if max_args == 0:
        sys.stderr.write("Usage: {path}\n".format(path=path))
    else:
        arg_list = list()
        optionals = 0
        
        for index in range(num_args):
            if index < min_args:
                arg_list.append(co.co_varnames[index])
            else:
                arg_list.append("[" + co.co_varnames[index])
                optionals += 1
                
        if max_args is None:
            arg_list.append("[" + co.co_varnames[num_args] + "")
            optionals += 1
            
        sys.stderr.write("Usage: {path} {args}{optional_closes}\n".format
                         (path=path,
                          args=" ".join(arg_list),
                          optional_closes="]" * optionals))
    if function.__doc__:
        sys.stderr.write("\n")
        sys.stderr.write(function.__doc__)
        sys.stderr.write("\n")
    
    return(1)

if __name__ == "__main__":
    sys.exit(__main__(main, sys.argv[0], sys.argv[1:]))