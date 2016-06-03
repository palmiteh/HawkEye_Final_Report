# DVIPDF  = dvipdft

# PDF	= $(SRC:%.tex=%.pdf)

# pdf: $(PDF)

# $(PDF): %.pdf: %.ps
# 	ps2pdf $<

LATEX	= pdflatex -shell-escape
DVIPS	= dvips
DVIPDF  = dvipdft
XDVI	= xdvi -gamma 4
GH		= gv

EXAMPLES = $(wildcard *.h)
SRC	:= $(shell egrep -l '^[^%]*\\begin\{document\}' *.tex)
TRG	= $(SRC:%.tex=%.dvi)
PSF	= $(SRC:%.tex=%.ps)
PDF	= $(SRC:%.tex=%.pdf)

pdf: $(PDF)

ps: $(PSF)

$(TRG): %.dvi: %.tex $(EXAMPLES)
	$(LATEX) $<
	$(LATEX) $<
	$(LATEX) $<

$(PSF):%.ps: %.dvi
	#$(DVIPS) -R -Poutline -t letter $< -o $@

$(PDF): %.pdf: %.ps
	#ps2pdf $<

show: $(TRG)
	@for i in $(TRG) ; do $(XDVI) $$i & done

showps: $(PSF)
	#@for i in $(PSF) ; do $(GH) $$i & done

writeup:
	pdflatex progress_report
	pdflatex progress_report
	pdflatex progress_report

text.pdf: progress_report
	ps2pdf progress_report

PDF: progress_report.pdf
	pdf progress_report.pdf

all: pdf

clean:
	rm -f *.pdf *.ps *.dvi *.out *.log *.aux *.bbl *.blg *.pyg *.toc

.PHONY: all show clean ps pdf showps

move:
	chmod a+r BetaReport.pdf
	cp BetaReport.pdf ~/public_html/
