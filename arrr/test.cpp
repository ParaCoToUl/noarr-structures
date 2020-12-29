
#include "arrr.hpp"

#include <iostream>
#include <typeinfo>
using namespace arrr;
using std::cout;
using std::endl;

/*TODO: convert this file to actual tests, also test negative test cases that
 * should cause a compilation failure (such as getting the level of a
 * nonexistent dimension). That may be pretty hard with normal test suites. */

int main() {
  auto a = scalar<float> ^ vector<'y'> ^ array<'x', 20>;
  cout << "has_dim tests:" << endl;
  cout << a % has_dim<'x'> << endl;
  cout << a % has_dim<'y'> << endl;
  cout << a % has_dim<'z'> << endl;
  cout << a % has_dims<'x'> << endl;
  cout << a % has_dims<'x', 'y'> << endl;
  cout << a % has_dims<'x', 'y', 'z'> << endl;

  auto as = a % resize<'y'>(30);  // TODO rest sized vectors

  cout << "more has_dim tests:" << endl;
  cout << as % has_dim<'x'> << endl;
  cout << as % has_dim<'y'> << endl;
  cout << as % has_dim<'z'> << endl;
  cout << as % has_dims<'x'> << endl;
  cout << as % has_dims<'x', 'y'> << endl;
  cout << as % has_dims<'x', 'y', 'z'> << endl;

  cout << "sizes:" << endl;
  cout << sizeof(a) << endl;
  cout << sizeof(as) << endl;
  cout << typeid(a).name() << endl;
  cout << typeid(as).name() << endl;

  cout << "levels/depths:" << endl;
  cout << a % depth << endl;
  cout << a % level<'x'> << endl;
  cout << as % level<'x'> << endl;
  cout << a % level<'y'> << endl;
  cout << as % level<'y'> << endl;

  auto asf = as % fix<'x'>(10);
  auto asff = asf % fix<'y'>(10);
  cout << "fixed index:" << asff % offset << endl;
  cout << sizeof(asf) << endl;
  cout << typeid(asf).name() << endl;
  cout << sizeof(asff) << endl;
  cout << typeid(asff).name() << endl;

  auto asffu = asff % unfix<'x'>;
  cout << sizeof(asffu) << endl;
  cout << typeid(asffu).name() << endl;

  auto asffua = asff % unfix_all;
  cout << sizeof(asffua) << endl;
  cout << typeid(asffua).name() << endl;

  cout << "multifix:" << endl;
  cout << as % fixs<'x', 'y'>(3, 5) % offset << endl;
  cout << as % idx<'x', 'y'>(3, 5) << endl;
  for (size_t i = 0; i < 10; ++i) cout << as % idx<'y', 'x'>(3, i) << endl;

  float x[20 * 30];
  cout << "actual data access:" << endl;
  cout << typeid(asff % at(x)).name() << endl;
  cout << asff % at(x) << endl;
  cout << "uninitialized read: " << *(asff % at(x)) << endl;
  *(asff % at(x)) = 10;
  cout << *(asff % at(x)) << endl;

  auto spl = scalar<char> ^ array<'X', 5> ^ array<'x', 4> ^
             bitblock21<'X', 'x', 2, 'x'>;

  cout << "split test:" << endl;
  for (size_t i = 0; i < 20; ++i) cout << spl % idx<'x'>(i) << endl;

  auto joi = scalar<char> ^ array<'z', 16> ^ zorder12<'z', 'x', 'y'>;

  cout << "join test:" << endl;
  for (size_t x = 0; x < 4; ++x) {
    for (size_t y = 0; y < 4; ++y) cout << joi % idx<'x', 'y'>(x, y) << '\t';
    cout << endl;
  }

  // TODO: test the normal modulo block
  auto magic = scalar<char> ^ array<'y', 4> ^ array<'x', 4> ^ array<'z', 16> ^
               zorder12<'z', 'Y', 'X'> ^ bitblock21<'Y', 'y', 2, 'y'> ^
               bitblock21<'X', 'x', 2, 'x'>;

  cout << "brutal magic test:" << endl;
  for (size_t x = 0; x < 16; ++x) {
    for (size_t y = 0; y < 16; ++y) cout << magic % idx<'x', 'y'>(x, y) << '\t';
    cout << endl;
  }

  cout << "tuple test:" << endl;
  auto tup1 = tuple<'a'> * scalar<float>;
  auto tup2 =
      tuple<'a'> * (scalar<float> ^ vector<'i'>)*scalar<int> ^ array<'j', 300>;

  cout << sizeof(tup1) << endl;
  cout << typeid(tup1).name() << endl;
  cout << typeid(tup2).name() << endl;
  cout << tup1 % size << endl;
  cout << tup2 % resize<'i'>(20) % size << endl;
  cout << tup2 % resize<'i'>(21) % size << endl;
  cout << tup2 % resize<'i'>(22) % size << endl;
  cout << sizeof(tup2) << endl;
  cout << sizeof(tup2 % fix<'i'>(123)) << endl;
  cout << sizeof(tup2 % fix<'i'>(123) % unfix_all) << endl;
  cout << tup2 % has_dim<'j'> << endl;
  cout << (tuple<'a'> * (scalar<float> ^ vector<'i'>)) % has_dim<'i'> << endl;

  cout << "tuple offsets:" << endl;
  cout << typeid(tup2 % field<'a', 0>).name() << endl;
  cout << typeid(tup2 % field<'a', 1>).name() << endl;
  auto t2s = tup2 % resize<'i'>(500);
  cout << t2s % field<'a', 0> % idx<'i', 'j'>(1, 2) << endl;
  cout << t2s % field<'a', 1> % idx<'j'>(1) << endl;

  auto t3 = tuple<'a'> * (tuple<'b'> * scalar<char> * scalar<int>)*(
                             tuple<'c'> * scalar<float> * scalar<double>);
  cout << t3 % fields<'a', 0, 'b', 0> % offset << endl;
  cout << t3 % fields<'a', 0, 'b', 1> % offset
       << endl;  // this gonna need some serious padding
  cout << t3 % fields<'a', 1, 'c', 1> % offset << endl;

  return 0;
}
