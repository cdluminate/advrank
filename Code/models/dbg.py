'''
Copyright (C) 2019-2020, Authors of ECCV2020 #2274 "Adversarial Ranking Attack and Defense"
Copyright (C) 2019-2020, Mo Zhou <cdluminate@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import os, sys
import traceback

def _bgcV(*msg) -> str:
    return f'\x1b[48;5;93m' + ' '.join(str(x) for x in msg) + '\x1b[m'

def _fgcG(*msg) -> str:
    return f'\x1b[38;5;42m' + ' '.join(str(x) for x in msg) + '\x1b[m'

def _fgcY(*msg) -> str:
    return f'\x1b[38;5;226m' + ' '.join(str(x) for x in msg) + '\x1b[m'

def _fgcGrey(*msg) -> str:
    return f'\x1b[38;5;240m' + ' '.join(str(x) for x in msg) + '\x1b[m'

def _fgcCyan(*msg) -> str:
    return f'\x1b[38;5;51m' + ' '.join(str(x) for x in msg) + '\x1b[m'

def printStack():
    '''
    Print the call stack. For debugging use.
    '''
    print('\x1b[38;5;161m', end='', file=sys.stderr)
    traceback.print_stack()
    print('\x1b[m', file=sys.stderr)

if __name__ == '__main__':
    print(_bgcV("_bgcV"))
    print(_fgcG("_fgcG"))
    print(_fgcGrey("_fgcGrey"))
