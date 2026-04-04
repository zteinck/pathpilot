from .base import FolderContents
from ....utils import is_file


class Files(FolderContents):
    '''
    Description
    --------------------
    Performs operations on all files as a unified group.
    '''

    #╭-------------------------------------------------------------------------╮
    #| Initialize Instance                                                     |
    #╰-------------------------------------------------------------------------╯

    def __init__(self, folder):
        super().__init__(folder)


    #╭-------------------------------------------------------------------------╮
    #| Instance Methods                                                        |
    #╰-------------------------------------------------------------------------╯

    def _to_list(self):
        return filter(
            # filter out lock files
            lambda x: x.name[:2] != '~$',
            filter(is_file, self.folder)
            )


    def delete(self):
        for file in self:
            file.delete()