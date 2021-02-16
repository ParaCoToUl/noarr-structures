# Mother of all requiremets
https://github.com/krulis-martin/cuda-kmeans

# Main ideas & Core principles
1. All GPU programs can be separated into: CPU logic/policies/GPU logic
2. Policy = logic of data transfering and accesing them on GPU.
3. Create high performace core for data modeling
4. Create library of common policies
5. Create binding into common languages like Python/R
6. Rewrite cuda-kmeans using new framework, to gain following:
  - Demontrate usability of our libraly
  - Hopefully achive simpler and shorter code of k-means
  - Natural support of diffener data layouts (easy swaps) (AOS vs SOA)
  - Similar or higher level of performance
7. Add fancy formating to this document
8. Get somehow at last mark 3 after we fail to achieve anything.
9. Run.

# Requiremets
**Functional requiremets**
- Support for C++, bindings to Python, R
- Predefined basic data layouts for quick ot of the box usage.

**Quality requiremets**
Usability
- dots








# Jirka wrote earlier (To be depricated):
Separate data layout from data access.
    Why: Algorithm implemementation should not depend on data layout.

Make the tool easy to use.
    => automatic return value typing
    => passing data slices to functions (to aid abstraction)
        HINT: define virtual data layouts that map to physical data layout?

Make the tool performant.
    => use compile-time features as much as possible
    => aviod pointer derefecencing when possible


Observations
------------

Tuples should not be iterated at runtime, causes return value ambiguity.
    RESULT: No, they have to be, because paths have to be runtime only.
    Otherwise we need to pass compiletime values as template arguments EVERYWHERE,
    even in the user code which makes it super messy.
    
    But that's ok, coz we infer the value type from virtual layout.

All algorithms at some point work with scalar values only - make scalar value access the priority.


# 16.2.2021 Notes
- Ty % už nesnáší i Mirek
- Martin, Mirek, Jirka# i já nelikujeme {} oproti () u resize
- Mirek byl vyděšenej z get_size{}
- | (or) je ok; % je úplně na ...; >> Je dost vytíženej. Mirkovi se nelíbí kvůli templatů; **Martin chce tečky. Mirek chce taky tečky, páč to je common case.** (Mirek nazval visual studio jako "shit"). **Mirek nepřekládá rozšířitelnost ohledně funktorů.** | a . jsou oboje velmi schůdný. **Závěr je takovej, že chceme tečky, low level dostupnej, ať si to lidi udělaj sami.**
- Smooth data jsou top.
- Jagged data: Mirek by se na ně zatím vykašlal. Martin: **Jagged data musí umět sekvenci různě velkejch blobů. To by mělo stačit.**
- Uděláme pokus na vektoru stringů až z toho vybublá co je potřeba supportovat.
## Liftování
- Ať nám hlavička nerozbíjí aligment
- Blob alignovanej na začátek stránky (HDF5) - Máme zachovat rozdělení na 2 fyly, tak jak to máme nějak
- Binární blob je super, hlavička zvlášť
- "Prostě to dodělejte"
- Do hlavičky metadata. Když budu chtít vlastní kompatní formát abych byl schopnej narvat vlastní hlavičku nebo appendovat do hlavičky
- **"Kdyby bylo .csv, tak bych učůrával blahem"**
## Python
- Mirek pošle nějakej browser, kterej to znásilní, ale bude to fungovat. Vzít libovolnej projekt (například nezaujatě k-means) a tam okoukat numpy
- Vytvořit demíčko, co to je. Trochu větší než volání funkce.
- (Mirek šel řešit píčovinu)
- Vysněnej interface: C-kový rozhraní, který je nějak anotovaný, který se přeloží do nějakýho interfacu. Který nějak zavolá tu naši pičovinu.
- Je rozumnej mezistupeň kompilace? Martinovi to je jedno. Mirek: "Je to jednoduchý a doporučuje to". Nějakym BASHem to je primitivní.
## R-ko
- Link: https://github.com/exaexa/scattermore
- Řádek 63, castění, C shit, pointer na int, scattermore, alfabelending, z toho kódu si příklad brát nemáme, "nebudu to radši komentovat", https://youtu.be/yR0EVtcPgD0
- 2 úrovně uživatelů: 1. je ten co v C napíše pičoviny, bashem vygeneruje kokotinu a tun si druhej uživatel strčí do src.
## Kernely jako lamda výrazy
- Long story shor: není to úplně core problému asi, takže na to spíš jebáme. Bylo by cool kdyby jsme mohli do budoucna top level používat dál. Mezivrstva/spodní je možná měnit.
- Defakto jsme došli k tomu, že to vlastně tak jako možná úplně jistě chceme. Ale ne nutně hned teď, včera to stačilo. Ale prej to není naše práce.
- Mirek: Spíš než framework by to měla bejt knihovna.
- Takže na to defakto teď sereme
- Tohle odsere někdo po nás

