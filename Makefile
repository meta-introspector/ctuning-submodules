IFS="\n"
#MARKDOWN=$(shell find . -iname "*.md")
MARKDOWN=$(shell cat markdown.txt)

# Form all 'html' counterparts
ORG=$(MARKDOWN:.md.org=.org)

.PHONY = all tar clean
all: $(ORG)

%.md.org: %.md
	pandoc --from markdown --to org $< -o $@

tar: $(MARKDOWN)
	tar --exclude=notes.tar.gz --exclude=.git/ -czvf notes.tar.gz ./

clean:
	rm $(ORG)


doindex:
	/mnt/data1/2023/10/24/emacs-data/vendor/emacs/org/microfts/microfts info -grams index  >debug.txt
