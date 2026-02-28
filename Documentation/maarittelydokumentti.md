## NumberReaderAI, Iivari van Uden - TKT
## Projektin kieli: Suomi

### Mitä ohjelmointikieltä käytät?
Käytän Pythonia.
### Kerro myös mitä muita kieliä hallitset siinä määrin, että pystyt tarvittaessa vertaisarvioimaan niillä tehtyjä projekteja.
Osaan myös C# ja GDScript. Hieman JS.
### Mitä algoritmeja ja tietorakenteita toteutat työssäsi?
Koodaan Neural Networkin. Siihen tarvitaan muun muassa Backpropagation ja cost function.
### Minkä ongelman ratkaiset?
Käsin kirjoitettujen numeroiden tunnistus.
### Mitä syötteitä ohjelma saa ja miten niitä käytetään?
Ohjelma saa syötteeksi grayscale kuvan käsin piirretystä numerosta, jonka treenattu neural network analysoi ja kertoo mikä numero on kyseessä.
### Tavoitteena olevat aika- ja tilavaativuudet (esim. O-analyysit)
Kun neuroverkkoa treenataan, on aikavaatiumus seuraava:
```
O(784*h + h² + h*10) * 2 * datan suuruus.
```
Jossa h on nodejen määrä layerissa.
### Lähteet, joita aiot käyttää.
3Blue1Brownin erittäin hyvin tehdyt videot neuroverkoista.
### Harjoitustyön Ydin.
Ohjelma pitää kouluttaa valmilla materiaalilla, että se voi analysoida uutta piirrettyä numeroa. Ohjelmaan siis syötetään kuva ja se kertoo, mikä kuva on kyseessä.

