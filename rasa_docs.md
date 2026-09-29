Rasa
Koncepti
	Kljucni fajlovi:
o	flows.yml: Definisu logicke korake za specificne situacije. Pogodni su u situacijama kada se definisu situacije za koje asistent treba da ima jasne korake izvrsavanja.
o	config.yml: Konfiurisu se dialogue understanding komponente I parametri, jezik i politike (npr. EnterpriseSearchPolicy I flows policy).
o	domain.yml: Definise univeryum – domen  za asistenta, koje sve akcije postoje, odgovori, slotovi…
o	actions.py: Sadrzi pajton kod za custom akcije, poput poziva koji su predvidjeni ka API-ju ili podesavanjima slotova.
o	endpoints.yml: Specifikacija ka svim eksternim serverima poput modela, baza za cuvanje info, nlg…
Davanje odgovora iz baze podataka:
Raspolozive baze:
	Faiss – in memory baza koja je zapravo integrisana u rasu, tj ubacuju se samo podaci u docs fajl. 
o	Prednost: Nema dodatnih troskova cuvanja fajlova, brza pretraga I odgovor.  
o	Mana: Svi fajlovi moraju biti .txt (potreno knvertovanje pdf u txt); vektorizacija se odvija u treningu, kao I pozivi ka embedding modelu -> dodavanjem novog fajla ponovo se vrsi vektorizacija u svih-> duzi trenig + placanje modela za iste stvari ponovo; ima smisla do 100 fajlova; samo vektorska pretraga, bez pretrage po kljucnim recima.
	Milvus/Qdrant – open source open-source vektorske baze. 
o	Prednost: skalira se na velike kolicine; kratka konfiguracija u rasa kodu; besplatno hostovanje za male velicine 
o	Mana: prihvataju samo vektore -> potrebno je izvrsiti vektorizaciju pdf-ova  pre ubacivanj(npr. pomocu LangChain-a) ili pomocu Embedding Function kod Milvus-a, tada pdf treba pretvoriti u txt I dodatno se placaju embeding modeli provajdera; ako je baza veliak, cena hostovanja je bas velika (bilo njihov Zilliz ili Qdrant Cloud ili nas lokalni Docker); samo vektorska pretraga
	Custom Information Retrieval with Enterprise Search Policy – ostale vektorske baze ili  indeks projavderi. Azure AI Search
o	Prednost: ubacivanje svih pdf iostalih dokumenata u azure doucment storage, na osnovu cega se pravi vektorski indeks; odlican kvalitet pretrage(hibrid + reranker); bezbednost zasigurana.
o	Mane: cena placanja baze podataka (do 50mb besplatno, posle toga planovi do 250e mesecno); potrebna posebna integraciona klasa za konekciju sa rasom; vezanost za Microsoft
-Potrebna konfiguracija (zavisi od vektorske baze koju krositimo):
1.	Konfiguracija za EnterpriseSearchPolicy u config.yml fajlu (definise se koji se store koristi, koja je model grupa za embedding,  koliko se dugo odogovr cuva  u istoriji caskanja, prompt za davanje odgovora, relevantnost – sta ako se odgovor ne nadje u bazi znanja). Komponenta SearchReadyLLMCommandGenerator za razumevanje dijaloga koja koristi EnterpriseSearch. 
2.	Endpoint ka bazi znanja u endpoints.yml (definisu se parametri za vector store(ako postoji) poput tipa, tacnog url, api kljuca, metatada-u…), kao i endpoint ka embedding modelu. 
3.	Definisanje pormptova za pretragu u prompts/enterprise_search_prompt.jinja2 rasa fajlu. 
4.	Definisanje utter odgovora u domain.yml, kao defoltnih odgovora na odredjene akcije poput situacije kada nema odgovora iz baze.
5.	defeinisanje paterna ponasanja koja prepustaju datu pretragu enterprise search-u u data/patterns.yml.

Pokretanje akcija koje menjaju nesto u bazi nase aplikacije
Postoje dva nacina:
1.	MCP funkcije
2.	Custom actions – preporuceno
Za pokretanje ovih akcija potrebno je prvo konfigurisati policy – FlowPolicy u config.yml fajlu. Nakon toga definisu se jasni flow-ovi I tacni koraci izvrsavanja svake od akcija. Na primer description, steps, actions, collect, link, kao I provere za slotove ili kraj toka.
Actions iz ovih tokova u zapravo custom akcije koje treba da se   implementiraju, a njihove implementacije se nalaze u actions.py. U rasa kodu, to su u sustini samo nacini na koji se salju info nasoj aplikaiciji, definisanje tacnog http get/post poziva, kao i definisanje sta su ocekivani rezultati.
MCP akcije tj funkcije pozivaju se kroz definisanje prvo parametara u endpoints.yml fajlu (koji je url, naziv, tip I ouath ili api key), a nakon toga se u flows.yml dodaje ime alata u sevreru, mcp_server: name – ime iz endpoint-a, I mapping za input I output ocekivane parametre. 

Chat history:
Cuvanje istorije razgovora obezbedjuje se konfiguracijom parametra tracker_store u endpoint.yml fajlu. Tracker store moze biti in-memory (to je I po default-u) I on radi do gasenja aplikacije, ali postoje I druge opcije poput SQLTrackerStore, RedisTrackerStore, MongoTrackerStore,  DynamoTrackerStore.
Problemi: gde cuvati sve te podatke, velicina koja se trosi zbog toga (do 1000 linija za 10ak poruka – jer rasa cuva svoje eventove I sve stepove kroz flow pri pozivu nekog zahteva). Za cuvanje samo poruka event_broker – ali bez tracker-a moze znaciti samo za prikaz na ui-u, jer llm nece dobijati kontekst odatle.
Sesija po korisniku obezbedjuje se tako sto se user_id prosledi iz aplikacije I cuva kao metapodatak koji u rasi ne moze da se promeni. 



