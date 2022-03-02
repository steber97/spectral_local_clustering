from __future__ import annotations

# Simple MergeFindSet implementation, useful to extract largest CC.
class MergeFindSet:
    def __init__(self, value) -> None:
        self.parent = self  # At the beginning, every mfs points to itself
        self.value = value
    
    def get_root(self):
        if self.parent == self:
            # This is the root
            return self
        else:
            # Update the parent, so that the complexity is O(1) in the average case.
            self.parent = self.parent.get_root()
            return self.parent
    
    def merge(self, mfs: MergeFindSet):
        if self.get_root() == mfs.get_root():
            # Do nothing, they are already merged
            pass
        else:
            # mfs's root now points to self.root
            mfs.get_root().parent = self.get_root()

    def same_mfs(self, mfs: MergeFindSet):
        return self.get_root() == mfs.get_root()