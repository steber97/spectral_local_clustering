using System.Collections.Generic;

namespace SubmodularHeatEquation
{
    /**
     * Simple implementation of a Merge Find Set, useful to compute the CCs of the hypergraph.
     */
    public class MergeFindSet
    {
        public MergeFindSet parent;
        public int value;

        public MergeFindSet(int value)
        {
            // At the beginning all nodes point to themselves.
            this.parent = this;
            this.value = value;
        }
        
        public MergeFindSet getRoot()
        {
            if (this.parent.value == this.value)
                return this;
            else
            {
                this.parent = this.parent.getRoot();
                return this.parent;
            }
        }

        public void merge(MergeFindSet mfs)
        {
            if (this.getRoot().value != mfs.getRoot().value)
                mfs.getRoot().parent = this.getRoot();
        }
    }
}