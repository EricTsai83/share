from io import StringIO
_branch_extend = '│  '
_branch_mid    = '├─ '
_branch_last   = '└─ '
_spacing       = ' ' * 3

def _getHierarchy(jsonData, name='', file=None, _prefix='', _last=True, head=True):
    """ Recursively parse json data to print data types """
    if head:
        print(_prefix, _branch_last if _last else _branch_mid, \
              name, sep="", file=file)
        
    _prefix += _spacing if _last else _branch_extend
    length = len(jsonData)
    for i, key in enumerate(jsonData.keys()):
        _last = i == (length - 1)
        _getHierarchy(jsonData[key], key, file, _prefix, _last)
        
def json2tree(jsonData, file=None, head=None):
    """ Output JSON data as tree to file or return as string """
    if file == None:
        messageFile = StringIO()
        _getHierarchy(jsonData, file=messageFile, head=False)
        message = messageFile.getvalue()
        messageFile.close()
        return message
    else:
        _getHierarchy(jsonData, file=file, head=False)
        
l = [['1'], ['2','3'], ['2','4'], ['2','5'], ['2','5','6'], ['2','5','7'], ['2','5','8'],['2','5','8','9'], ['10'],['11','12']]
root = {}
for path in l:
    parent = root
    for n in path:
        parent = parent.setdefault(n, {})
        
print(json2tree(root))
