from .base import FolderContents
from ...utils import is_folder


class Subfolders(FolderContents):
    '''
    Description
    --------------------
    Performs operations on all subfolders as a unified group.
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
        return filter(is_folder, self.folder)


    def delete(self):
        for folder in self:
            folder.delete()
        self.folder._clear_subfolder_cache()