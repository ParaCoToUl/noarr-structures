if __name__ != "__main__":
	raise Exception('do not import me')

import sys, os, os.path
import re; Pattern = re.compile

from config import global_decls, substitutions, convert_synopsis

###############
# Definitions #
###############

scope_block = lambda f: f()

snippatt = Pattern('\n```([a-z]+)\n(.*?\n)```\n', re.DOTALL)
headpatt = Pattern('(\n+)(#+) ([^\n]+)(?=(\n+))', re.DOTALL)
tickpatt = Pattern('`.*?`')
linkpatt = Pattern(r'\[(.*?)\]\((.*?)\)')
cudapatt = Pattern('(.*)<<<(.*)>>>(.*)')

anchor_ign_patt = Pattern('[(),.:<>`]')

def clean_anchor(text):
	return anchor_ign_patt.sub('', text).replace(' ', '-').lower()
def clean_anchor_1(text):
	return anchor_ign_patt.sub('', text).replace(' ', '')

def cuda_replacement(mo):
	kernel, params, rest = mo.group(1, 2, 3)
	return f'void UNIQ() {{ snippet_check_kernel_params({params}); {kernel}{rest} }}'

##################
# Initialization #
##################

try:
	[cmd, snipdir, include, *args] = sys.argv
	if not args: raise Exception
except:
	print('Usage:', cmd, '<snippets output dir> <include dir> <markdown files ...>')
	sys.exit(1)

if os.listdir(snipdir):
	print(snipdir, '(snippet output dir) is not empty')
	sys.exit(1)

available = set()
linked = []
snipfns = 0

@scope_block
def _():
	inc = include
	if not include.endswith('/'):
		inc += '/'
	def rel_include(dn):
		if not dn.startswith(inc):
			raise Exception(dn, 'does not start with', inc)
		return dn[len(inc):]
	with open(f'{snipdir}/snippet_check_all.hpp', 'x') as f:
		headers = sorted(f'{rel_include(dn)}/{fn}' for dn, _, fns in os.walk(inc) for fn in fns)
		if not headers:
			print('Include directory empty or missing')
			exit(1)
		for h in headers:
			if not h.endswith('.hpp'):
				continue
			elif h.endswith('/omp.hpp') or h.endswith('/std_para.hpp'):
				continue
			f.write(f'#include <{h}>\n')
		f.write(global_decls)
		f.write('#define UNIQ_A(CTR) UNIQ_##CTR\n')
		f.write('#define UNIQ_B(CTR) UNIQ_A(CTR)\n')
		f.write('#define UNIQ UNIQ_B(__COUNTER__)\n')
		f.write('extern struct snippet_check_ret {} snippet_check_ret;\n')
		f.write('static void snippet_check_kernel_params(dim3, dim3) {}\n')

########
# Main #
########

for file in args:
	with open(file, newline='') as f:
		content = f.read()

	##########################
	# Basic file type checks #
	##########################
	if not content:
		print(file, 'is empty')
		continue
	if '\ufeff' in content:
		print(file, 'contains BOM')
		continue
	if '\r' in content:
		print(file, 'contains CR')
		continue
	if content[-1] != '\n':
		print(file, 'last line misses LF')
		continue

	#######################################
	# Separate the snippets from the text #
	#######################################
	snippets = [] # tuples of (snippet index in file, byte offset in file, c++ source)
	template_snippets = [] # same format, but they go in a namespace scope instead of fn scope
	def replacement(mo):
		idx = len(snippets) + len(template_snippets)
		code = mo.group(2)
		match mo.group(1):
			case 'cpp':
				coll = template_snippets if 'template<' in code else snippets
			case 'cu':
				code = cudapatt.sub(cuda_replacement, code.replace('__global__', ''))
				coll = template_snippets if 'template<' in code else snippets
			case 'hpp' | 'cuh':
				code = convert_synopsis(code)
				if not code:
					print(file, 'contains unparsable synopsis')
				coll = template_snippets
			case _:
				print(file, 'contains unrecognized snippet type', mo.group(1))
				coll = [] # append to nowhere
		coll.append((idx, mo.start(2), code))
		return '\n<!-- BLOCK SNIPPET -->\n'
	text = snippatt.sub(replacement, content)

	######################################
	# Write out snippets for compilation #
	######################################
	if snippets:
		@scope_block
		def _():
			with open(f"{snipdir}/{file.replace('/', '%')}.cpp", 'x') as f:
				f.write('#include "snippet_check_all.hpp"\n\n')
				f.write(f'struct snippet_check_ret &snippet_check_fn_{snipfns}()\n')
				for idx, off, code in snippets:
					for p, r in substitutions.get((file, idx), {}).items():
						code = re.sub(p, r, code, flags=re.MULTILINE)
					line = content[:off].count('\n') + 1
					f.write('{\n')
					f.write(f'#line {line} "{file}" // {idx}\n')
					f.write(code)
					f.write(f'#line {1000*idx+1} "GENERATED"\n')
					f.write('\n')
				f.write(f'return snippet_check_ret;\n')
				for _ in snippets:
					f.write('}\n')
		snipfns += 1
	if template_snippets:
		@scope_block
		def _():
			with open(f"{snipdir}/{file.replace('/', '%')}.tmpl.cpp", 'x') as f:
				f.write('#include "snippet_check_all.hpp"\n\n')
				f.write('namespace {\n')
				for idx, off, code in template_snippets:
					for p, r in substitutions.get((file, idx), {}).items():
						code = re.sub(p, r, code, flags=re.MULTILINE)
					line = content[:off].count('\n') + 1
					f.write('namespace UNIQ {\n')
					f.write(f'#line {line} "{file}" // {idx}\n')
					f.write(code)
					f.write(f'#line {1000*idx+1} "GENERATED"\n')
					f.write('\n')
				for _ in template_snippets:
					f.write('}\n')
				f.write('} // unnamed namespace\n')

	#############################################################
	# There should be exactly one H1, located on the first line #
	#############################################################
	if not text.startswith('# '):
		print(file, 'does not start with a H1')
		continue
	if '\n# ' in text:
		print(file, 'contains stray H1')

	#################################
	# The H1 should match file name #
	#################################
	@scope_block
	def _():
		h1len = text.index('\n')
		if text[h1len:][:2] != '\n\n':
			print(file, 'H1 not followed by empty line')
		if text[h1len+2] == '\n':
			print(file, 'H1 followed by too many empty lines')
		if clean_anchor_1(text[2:h1len]) + '.md' != file.split('/')[-1]:
			print(file, 'H1 does not match filename')

	##########################
	# H1 only without anchor #
	##########################
	available.add(file)

	####################################
	# Check and collect other headings #
	####################################
	@scope_block
	def _():
		headings = headpatt.findall(text)
		if len(headings) != text.count('\n#'):
			print(file, 'contains unmatched heading (bug in regex?)')
		for nl_before, hashes, htext, nl_after in headings:
			level = len(hashes)
			if level < 2 or level > 5:
				print(file, 'contains suspicious heading: level', level)
			if len(nl_after) != 2:
				print(file, 'contains bad num of blank lines after', htext)
			if len(nl_before) != (3 if level==2 else 2):
				print(file, 'contains bad num of blank lines before', htext)
			available.add(file + '#' + clean_anchor(htext))

	###########################
	# Check other blank lines #
	###########################
	if '\n\n\n' in text.replace('\n\n\n#', ''):
		print(file, 'contains doubled blank lines (not followed by a heading)')

	#############################################################
	# Remove inline snippets so they don't interfere with links #
	#############################################################
	text = tickpatt.sub('<!-- SNIPPET -->', text)
	if '`' in text:
		print(file, 'contains stray backtick or unmatched snippet')
		continue

	##########################################################
	# Remember links (check later, after collecting targets) #
	##########################################################
	@scope_block
	def _():
		for link in linkpatt.finditer(text):
			(full, lntext, target) = link.group(0, 1, 2)
			if (
				lntext.count('(') != lntext.count(')') or
				'[' in lntext or ']' in lntext or
				'(' in target or ')' in target or
				'[' in target or ']' in target):
				print(file, 'contains corrupted link', full)
			is_img = text[link.start() - 1] == '!'
			linked.append((file, target, is_img))

###############
# Check links #
###############
@scope_block
def _():
	for source, target, is_img in linked:
		if target.startswith('https://'):
			continue

		if target.startswith('#'):
			if is_img:
				print('Invalid image linking to anchor')
				continue
			resolved = source + target
		else:
			left = source.split('/')
			left.pop()
			right = target
			while right.startswith('../'):
				left.pop()
				right = right[3:]
			left.append(right)
			resolved = '/'.join(left)
			del left, right

		if is_img: # file
			if not os.path.isfile(resolved):
				print('Unresolved image', dict(source=source, relative=target, absolute=resolved))
		else: # markdown link
			if not resolved in available:
				print('Unresolved', dict(source=source, relative=target, absolute=resolved))

###################
# Main executable #
###################
@scope_block
def _():
	with open(f'{snipdir}/main.cpp', 'x') as f:
		f.write('#include <stdlib.h>\n')
		f.write('struct snippet_check_ret {} snippet_check_ret;\n')
		f.write('int main() {\n')
		for i in range(snipfns):
			f.write(f'extern struct snippet_check_ret &snippet_check_fn_{i}();\n')
			f.write(f'if(&snippet_check_fn_{i}() != &snippet_check_ret) abort();\n')
		f.write('return 0;\n')
		f.write('}\n')
