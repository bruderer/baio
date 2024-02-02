import gffutils


###building a database from a GFF file
db = gffutils.create_db('genome/final_annotation_sorted_fixed.gff', dbfn='./genome/gff.db', force=True, keep_order=True, merge_strategy='merge', sort_attribute_values=True)
