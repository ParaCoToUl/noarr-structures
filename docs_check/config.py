_PROLOG = r'\A'
_EPILOG = r'\Z'
_ANY = '(?:.|\n)'

_tmp_mulsc = '''
auto sc = noarr::scalar<float>() ^ noarr::sized_vector<'i'>(42) ^ noarr::sized_vector<'j'>(42);
auto sr = noarr::scalar<float>() ^ noarr::sized_vector<'j'>(42) ^ noarr::sized_vector<'i'>(42);
auto bc = noarr::make_bag(sc, (void*)nullptr);
auto br = noarr::make_bag(sr, (void*)nullptr);
'''
_tmp_x = "noarr::array_t<'x', 42, noarr::scalar<float>>::signature"
_tmp_y = _tmp_x.replace('x', 'y')

global_decls = '''
#include <../tests/noarr_test_cuda_dummy.hpp>
#include <noarr/structures/interop/cuda_striped.cuh>
#include <noarr/structures/interop/cuda_traverser.cuh>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
'''

substitutions = {
	('docs/BasicUsage.md', 1): {'auto': ''},
	('docs/BasicUsage.md', 16): {'n(rows|cols)': '42'},
	('docs/BasicUsage.md', 17): {_EPILOG: _tmp_mulsc + 'void UNIQ() { mul_by_scalar(sr, nullptr, 4.2); mul_by_scalar(sc, nullptr, 4.2); }\n'},
	('docs/BasicUsage.md', 18): {_EPILOG: _tmp_mulsc + 'void UNIQ() { mul_by_scalar(br, 4.2); mul_by_scalar(bc, 4.2); }\n'},
	('docs/BasicUsage.md', 19): {_EPILOG: _tmp_mulsc + 'void UNIQ() { mul_by_scalar(br, 4.2); mul_by_scalar(bc, 4.2); }\n'},
	('docs/DefiningStructures.md', 5): {'image': 'UNIQ'},
	('docs/DefiningStructures.md', 6): {'using signature = \.\.\.;': '', '\.\.\.': '0'},
	('docs/DimensionKinds.md', 0): {_PROLOG: 'using noarr::lit;', '.*incorrect.*': ''},
	('docs/DimensionKinds.md', 4): {'.*fails at compile time.*': ''},
	('docs/DimensionKinds.md', 5): {'.*fails at compile time.*': '', '// option': '{//', '// \.\.\.': '} } } }'},
	('docs/Signature.md', 9): {'/\*\.\.\.\*/::signature': _tmp_x, _EPILOG: f'static_assert(std::is_same_v<new_sig, {_tmp_y}>);\n', 'WANT_DEAD_CODE': '1'},
	('docs/State.md', 0): {'1<<i': '1'},
	('docs/State.md', 2): {'my_state': 's1'},
	('docs/State.md', 3): {'my_state': 's1'},
	('docs/Traverser.md', 6): {'.*this fails.*': ''},
	('docs/Traverser.md', 7): {'.*this fails.*': ''},
	('docs/Traverser.md', 8): {'(from|a|b)_data': 'std::calloc(400*500, sizeof(float))'},
	('docs/Traverser.md', 13): {'#pragma omp parallel for': ''},
	('docs/Traverser.md', 14): {'#pragma omp parallel for': ''},
	('docs/Traverser.md', 18): {
		'matrix_data': 'std::calloc(300*400, sizeof(float))',
		f'(noarr::tbb_reduce{_ANY}*,)({_ANY}*)noarr::tbb_reduce.*,': lambda mo: ''.join(mo.group(1, 2, 1)),
	},
	('docs/Traverser.md', 19): {_PROLOG: 'void *values_data = nullptr; std::size_t size = 0;'},
	('docs/Traverser.md', 20): {'[ab]_data': '(void*)nullptr'},
	('docs/Traverser.md', 21): {'[ab]_data': '(void*)nullptr'},
	('docs/other/Functions.md', 0): {_PROLOG: "auto matrix = noarr::scalar<int>() ^ noarr::sized_vector<'x'>(1);", '.*(will not work|not make sense).*': ''},
	('docs/other/Functions.md', 1): {_PROLOG: "auto matrix = noarr::scalar<int>() ^ noarr::sized_vector<'x'>(1);"},
	('docs/other/Mangling.md', 0): {'/\*\.\.\.\*/': '0'},
	('docs/other/SeparateLengths.md', 2): {'.*get_length.*': ''},
	('docs/other/Serialization.md', 1): {'/\*\.\.\.\*/': 'noarr::tuple_t<\'*\'>()', 'path/to/(src|dest)': '/dev/null', 'return 1': 'std::abort()'},
	('docs/other/StructureTraits.md', 0): {'State = state<>': 'State = noarr::state<>', '/\*\.\.\.\*/': 'void'},
	('docs/structs/array.md', 0): {'using array = .*;': 'struct array_t;'},
	('docs/structs/cuda_step.md', 1): {'cuda_step_grid\(\)': 'step(0, 1024*1024)'},
	('docs/structs/cuda_step.md', 2): {'cuda_step_block\(\)': 'step(0, 1024*1024)'},
	('docs/structs/cuda_step.md', 3): {"cuda_step_block<'j'>\(\)": "step<'j'>(0, 1024*1024)"},
	('docs/structs/cuda_step.md', 4): {"cuda_step_grid<'t'>\(\)": "step<'t'>(0, 1024*1024)"},
	('docs/structs/into_blocks.md', 2): {'/\*\.\.\.\*/': '[](auto){}'},
	('docs/structs/into_blocks.md', 5): {'num_elems': '42', 'input_data': 'std::calloc(42, sizeof(float))'},
	('docs/structs/into_blocks.md', 6): {'num_elems': '42', 'input_data': 'std::calloc(42, sizeof(float))'},
	('docs/structs/merge_zcurve.md', 0): {', char Dim': ''},
	('docs/structs/rename.md', 1): {'/\*\.\.\.\*/': 'std::calloc(42, sizeof(float))'},
	('docs/structs/rename.md', 2): {'/\*\.\.\.\*/': '(void*)nullptr', '^matmul': 'auto UNIQ = (matmul', '^\)': '),0)'},
	('docs/structs/rename.md', 3): {'/\*\.\.\.\*/': '(void*)nullptr'},
	('docs/structs/vector.md', 0): {'struct vector;': 'struct vector_t;'},
	('docs/structs/slice.md', 6): {_PROLOG: '#if __cplusplus >= 202002L\n', 'lit': 'noarr::lit', _EPILOG: '#endif\n'},
	('docs/structs/tuple.md', 0): {'/\*\.\.\.\*/': 'noarr::scalar<int>'},
	('docs/structs/tuple.md', 4): {'num_edges': '1024', 'data_ptr': '(void*)nullptr'},
}

def convert_synopsis(code):
	match code.replace('noarr::', '').split('\n', 2):
		case [first, '', rest] if first.startswith('#include'):
			# the synopsis can use auto in parameters list - do not check it against c++17
			first = '#if __cplusplus >= 202002L'
			xfirst = '#endif'
			# we don't want the definitions in scope for the next snippet - throw them away immediately
			second = 'namespace UNIQ { typedef void proto;'
			xsecond = '}'
			#
			return '\n'.join([first, second, rest, xsecond, xfirst, ''])
		case _:
			return ''
