find -name \*.org | xargs grep -h . | sort -u > ulines.txt
find -name \*.org | xargs grep -h . | sort |uniq -c | sort -n > lines.txt
tr -s '[[:punct:][:space:]]' '\n'  < ulines.txt  > words.txt
sort uwords.txt |uniq -c | sort -n >wordcounts.txt
