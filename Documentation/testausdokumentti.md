## Testausdokumentti

estausdokumentin pitääs sisältää seuraavat:

Yksikkötestauksen kattavuusraportti.

### Yksikkötestauksen kattavuusraportti
![coverage](coverage.png)

### Mitä on testattu, miten tämä tehtiin?
Testit tarkistavat, että:
- Kuvien data saadan ongelmitta.
- Neuroverkon rakenne on oikea ja vastaus seuraa sitä.
- Kahden samoilla painolla luodut neuroverkot antavat samat tulokset.
- Neuroverkkoa treenatessa painot muuttuvat ja cost pienenee.

### Minkälaisilla syötteillä testaus tehtiin?
Testeille luotiin omat neuroverkot, jotka olivat paljon pienempiä. Painot olivat satunnaisia.ö

### Miten testit voidaan toistaa?
```
poetry run coverage run --branch -m pytest tests
```