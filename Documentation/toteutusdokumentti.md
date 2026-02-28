## Ohjelman yleisrakenne
Ohjelma on jaettu network.py, data_handler.py ja interface.py.
Data_handler käsittelee ja hakee tietokannasta dataa, kun taas network käsittelee kaiken, joka liittyy neuroverkkoon.
Interface yhdistää kummatkin ja luo käyttöliittymän.
## Saavutetut aika- ja tilavaativuudet (esim. O-analyysit pseudokoodista)
Kun neuroverkkoa treenataan, on aikavaatiumus seuraava:
```
O(784*h + h² + h*10) * 2 * datan suuruus.
```
Jossa h on nodejen määrä layerissa.
## Työn mahdolliset puutteet ja parannusehdotukset
Haluaisin luoda nettisivun, jolla käyttäjä voi piirtää itse numeron, joka syötetään neuroverkkoon.
## Laajojen kielimallien (ChatGPT yms.) käyttö. Mainitse mitä mallia on käytetty ja miten. Mainitse myös mikäli et ole käyttänyt. Tämä on tärkeää!
Olen käyttänyt ChatGPT:tä selittämään tarkemmin neuroverkkojen käsitteitä, jotka eivät selvenneet minulle videoista tai lähteistä. Pääosin Backpropagation. Käytin myös ChatGPT:tä selittämään minulle, miten numpy toimii ja miten sen tähän projektiin oleellisimmat komennot toimivat (dot, zip, outer, npz).
Viimeiseksi olen käyttänyt ChatGPT:tä selventämään neuroverkkojen monimutkaisia matemaattisia kaavoja, taas pääosin backpropagationissa. 
## Lähteet, joita olet käyttänyt, vain ne joilla oli merkitystä työn kannalta.
[3Blue1Brownin neuroverkko sarja](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=)
[Samson Zhang:in video neuroverkon koodaamisesta](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=4s)