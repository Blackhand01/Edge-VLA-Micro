# Jetson Orin Nano Development Setup Guide
**File Name:** `jetson-install-tutorial.md`  
**Version:** 1.0.0  
**Date:** 29 Maggio 2026  
**Author:** `[REDACTED]`  
**Target Hardware:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  
**Host Environment:** PC x86_64 con Ubuntu 22.04 LTS nativo  
**Client Environment:** Mac con macOS per sviluppo headless via VS Code  

---

## 1. Introduzione e Architettura del Sistema
Questa guida documenta l'intera procedura end-to-end per la configurazione, il flashing e l'impostazione di un ambiente di sviluppo "headless" avanzato per il **NVIDIA Jetson Orin Nano Developer Kit**. 

L'obiettivo è ottenere un sistema di sviluppo flessibile e ad altissime prestazioni per l'ottimizzazione dell'inferenza e l'ingegneria dei sistemi, strutturato come segue:
* **Target (Jetson Orin Nano):** Esegue **JetPack 6.2.2** (basato su Ubuntu 22.04 LTS), installato nativamente su un **SSD NVMe Crucial** ad alta velocità per azzerare i colli di bottiglia di I/O.
* **Host (PC Linux x86_64):** Configurato in dual boot nativo (*bare-metal*) con **Ubuntu 22.04 LTS**. Viene utilizzato esclusivamente come ponte hardware per effettuare il flashing a basso livello tramite l'SDK Manager di NVIDIA.
* **Client (Mac):** La postazione di lavoro principale dal quale si scrive codice, si compilano moduli e si esegue il debug da remoto tramite sessioni SSH e Visual Studio Code.

### Diagramma Logico delle Connessioni

```

```text
File jetson-install-tutorial.md generato con successo.


```

```
 +---------------------------------------------------------+
 |                    Router Wi-Fi                         |
 +------------+-------------------------------+------------+
              | (192.168.1.x)                 | (192.168.1.xxx)
              v                               v
     +-----------------+             +-----------------+
     |       Mac       |             | Jetson Orin Nano|
     |    (Client)     |             |    (Target)     |
     +--------+--------+             +--------+--------+
              |                               |
              +--- Cavo USB-C (Ad alta vel.) -+
                   Interfaccia Linux for Tegra
                   (IP: 192.168.55.100 <-> 192.168.55.1)

```

```

---

## 2. Requisiti Hardware e Preparazione Componenti

Prima di iniziare, assicurarsi di avere a disposizione i seguenti componenti hardware:
1.  **NVIDIA Jetson Orin Nano Dev Kit (8GB):** Il modulo e la relativa carrier board.
2.  **SSD NVMe Crucial (es. 500GB/1TB):** Rimosso dall'adattatore USB esterno e installato direttamente nello **slot M.2 Key M** situato sul fondo della carrier board del Jetson (sotto il modulo principale).
3.  **Cavo USB-C dati ad alta velocità:** Utilizzare il cavo schermato in dotazione con l'adattatore NVMe Crucial. *Nota: Evitare cavi di ricarica standard per smartphone, poiché spesso sono privi delle linee dati interne.*
4.  **Strumenti per il Cortocircuito (Jumper):** Una graffetta metallica sagomata a "U", una pinzetta da elettronica o un cacciavite a taglio di precisione.

![Mappa Hardware del Jetson Orin Nano e Slot M.2 Inferiore](https://developer.nvidia.com/sites/default/files/akamai/embedded/images/jetson-orin-nano-dev-kit.png)
*(Placeholder Immagine: Posizionamento dello slot NVMe sotto la scheda madre del Jetson)*

---

## 3. Fase 1: Configurazione del PC Host Linux

L'utilizzo di un sistema operativo nativo (bare-metal) è **obbligatorio**. Le macchine virtuali (VM) o WSL incontrano gravi problemi di passthrough USB e di allocazione dei driver di rete durante la delicata fase di scrittura del bootloader del Jetson.

### 3.1 Risveglio del Dual Boot e Connessione a Internet
Se la partizione Ubuntu 22.04 LTS del PC host non viene utilizzata da tempo o presenta problemi di connettivita, è necessario portarla online per scaricare i pacchetti ufficiali NVIDIA.

* **Soluzione di Emergenza (Tethering USB da Smartphone):** Collegare uno smartphone tramite cavo USB al PC host. Andare nelle impostazioni del telefono ed attivare **Hotspot e tethering > Tethering USB**. Ubuntu riconoscerà istantaneamente il telefono come una connessione di rete cablata Ethernet (es. `usb0`) senza bisogno di driver aggiuntivi.

### 3.2 Aggiornamento dei Repository e del Sistema
Aprire il terminale del PC host (`Ctrl + Alt + T`) ed eseguire la pulizia e l'allineamento dei pacchetti:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt autoremove -y

```

---

## 4. Fase 2: Installazione di NVIDIA SDK Manager sull'Host

L'**NVIDIA SDK Manager** è l'ambiente centralizzato per scaricare JetPack, i driver della scheda (BSP) e tutti i toolkit di sviluppo (CUDA, TensorRT).

1. Navigare sul sito [NVIDIA Developer SDK Manager](https://www.google.com/search?q=https://developer.nvidia.com/embedded/sdk-manager).
2. Effettuare il login con un account Developer gratuito.
3. Scaricare il pacchetto dedicato: **Download for Ubuntu (.deb x86_64)**.
4. Installare il file scaricato risolvendo le dipendenze tramite `apt`:
```bash
cd ~/Scaricati
sudo apt install ./sdkmanager_*_amd64.deb

```


5. Avviare l'applicazione da terminale o dai programmi di Ubuntu:
```bash
sdkmanager

```



---

## 5. Fase 3: Cablaggio e Modalità di Recupero Hardware (Force Recovery Mode)

Per consentire all'Host di formattare e scrivere direttamente sulla memoria NVMe del Jetson, la scheda deve essere avviata in una modalità speciale denominata **Force Recovery Mode (FCM)**.

### 5.1 Correzione del Cablaggio USB

Il cavo USB-C ad alta velocità proveniente dal PC host **deve essere inserito esclusivamente nella porta USB-C nativa del Jetson** (posizionata sul retro della scheda, tra il jack di alimentazione e la porta Ethernet).

* *Errore comune:* Collegare il PC Host a una delle quattro porte USB-A (rettangolari). Quelle porte funzionano solo in modalità *Host* (per mouse, tastiere e fotocamere) e non accettano il flashing.

### 5.2 Sequenza Temporale del Cortocircuito Hardware (Pin Jumper)

Poiché il Dev Kit Orin Nano non dispone di pulsanti fisici per la modalità di recupero, è necessario agire sulla morsettiera dei pin scoperti (*Button Header* a 12 pin posizionato sotto il blocco del dissipatore).

1. **Scollegare completamente l'alimentazione** dal Jetson (il LED verde sulla scheda deve essere spento). Il cavo USB-C verso il PC host può rimanere inserito.
2. Individuare i pin **FC REC** (Force Recovery) e **GND** (Ground), corrispondenti ai **pin 9 e 10** del Button Header (fare riferimento alle scritte serigrafate sul circuito stampato sul fondo della scheda).
3. Utilizzare una graffetta metallica piegata a "U" o una pinzetta per **mettere saldamente in corto** i pin 9 e 10, assicurando un contatto elettrico perfetto.
4. **Mantenendo il corto**, inserire lo spinotto rotondo dell'alimentatore (Jack DC 9V-19V). Il LED verde del Jetson si accenderà.
5. Mantenere il corto in posizione per **5 secondi completi** dopo l'accensione del LED, dopodiché rimuovere lo strumento (graffetta o pinzetta).

```
   Mappa concettuale dei Pin (Button Header):
   +-----------------------------------------+
   |  o   o   o   o   o   o   o   [o]  [o]  o |  <- Fila superiore pin
   |  1   2   3   4   5   6   7    9    10  11|
   |                              |    |     |
   |                              +----+-----+
   |                                Corto (FCM)
   +-----------------------------------------+

```

### 5.3 Verifica dello Stato di Recovery sull'Host

Aprire un terminale sul PC host e digitare:

```bash
lsusb

```

Se il cortocircuito è avvenuto con il giusto tempismo, nell'elenco deve comparire una riga contenente l'ID del produttore NVIDIA:
`Bus XXX Device YYY: ID 0955:7035 NVIDIA Corp.`

---

## 6. Fase 4: Configurazione e Flashing tramite SDK Manager

Una volta che l'hardware è correttamente esposto in modalità APX/Recovery, l'interfaccia dell'SDK Manager mostrerà una finestra di dialogo di auto-rilevamento.

### 6.1 Step 01: Selezione del Prodotto

* **Target Hardware:** Scegliere dal menu a tendina **Jetson Orin Nano [8GB developer kit version]** (Modulo P3767-0005, Carrier Board P3768-0000). *Non selezionare la versione standard "8GB" senza la dicitura dev kit, altrimenti il partizionamento fallirà.*
* **Target Operating System:** Selezionare l'ultima versione stabile, **JetPack 6.2.2** (Linux for Jetson).
* Cliccare su **Continue to Step 02**.

### 6.2 Step 02: Selezione dei Componenti e Licenze

Per ottimizzare i tempi di download ed evitare di saturare lo spazio sul PC host, configurare i flag come segue:

* **HOST COMPONENTS:** **Deselezionare completamente**. I tool di sviluppo sul PC host non sono necessari, poiché l'intera pipeline girerà remotata sul Mac.
* **TARGET COMPONENTS:** * `Jetson Linux` (Immagine del sistema operativo e script di Flash) -> **Selezionato**.
* `Jetson Runtime Components` (Librerie di esecuzione base) -> **Selezionato**.
* `Jetson SDK Components` (CUDA Toolkit, cuDNN, TensorRT, OpenCV) -> **Selezionato obbligatoriamente**. *Senza questi componenti non sarà possibile usufruire dei compilatori e degli header per l'ottimizzazione dell'inferenza a basso livello.*


* Spuntare la casella delle licenze: *"I accept the terms and conditions..."*.
* Cliccare su **Continue to Step 03**.

### 6.3 Destinazione dello Storage (Crucial NVMe)

Subito dopo l'avvio, l'SDK Manager aprirà una finestra di configurazione OEM (Storage Device).

* **Hardware di Destinazione:** Cambiare l'impostazione predefinita (SD Card o eMMC) selezionando espressamente **NVMe**.
* Fornire la password di root di Ubuntu del PC host quando richiesto.
* Il sistema scaricherà circa 15-20 GB di dati, compilerà l'immagine e la flasherà direttamente sull'SSD Crucial tramite il cavo USB-C. Al termine dell'operazione, il Jetson si riavvierà automaticamente avviando il nuovo sistema operativo.

---

## 7. Fase 5: Configurazione di Rete e Wi-Fi sul Jetson

Al primo avvio, completare la procedura guidata di Ubuntu sul Jetson (creazione utente, impostazione password e nome host). Il Jetson Orin Nano dispone di un modulo Wi-Fi integrato (AW-CB375NF) posizionato vicino all'SSD NVMe.

### 7.1 Connessione alla Rete Wi-Fi da Terminale Nativo

Per agganciare il Jetson alla rete domestica senza l'ausilio di interfacce grafiche, utilizzare lo strumento di rete `nmcli` da terminale:

```bash
sudo nmcli device wifi connect "<SSID_WIFI>" password "<PASSWORD_WIFI>"

```

L'output confermerà l'attivazione: `Device 'wlP1p1s0' successfully activated...`

### 7.2 Ispezione degli Indirizzi IP Assegnati

Eseguire il comando di verifica delle interfacce:

```bash
hostname -I

```

L'output mostrerà molteplici indirizzi IP, fondamentali per la successiva fase di controllo remoto:

1. **`192.168.1.xxx` (Wi-Fi locale):** L'IP dinamico assegnato dal router.
2. **`192.168.55.1` (USB Device Mode):** Un IP statico nativo generato dal modulo *Linux for Tegra*. Quando il Jetson è collegato a un computer tramite cavo USB-C, emula una scheda di rete cablata fissa a questo indirizzo.
3. **`172.17.0.1` (Docker Bridge):** L'interfaccia virtuale preconfigurata di Docker.

---

## 8. Fase 6: Configurazione del Controllo Remoto da Mac

Da questo momento in poi, il PC host può essere spento. Tutto lo sviluppo avverrà dal Mac sfruttando la connessione SSH.

### 8.1 Primo Collegamento e Scambio delle Chiavi Crittografiche

Aprire il **Terminale di sistema di macOS** sul Mac e lanciare la connessione verso l'IP Wi-Fi del Jetson:

```bash
ssh <UTENTE_JETSON>@192.168.1.xxx

```

Il sistema risponderà bloccando la connessione per sicurezza:
`The authenticity of host '192.168.1.xxx' can't be established. Are you sure you want to continue connecting (yes/no/[fingerprint])?`

1. Digitare **`yes`** e premere Invio per salvare l'impronta ED25519 nel file `known_hosts` del Mac.
2. Inserire la password dell'utente configurato sul Jetson.
3. Il terminale mostrerà il banner di benvenuto: `Welcome to Ubuntu 22.04.5 LTS`.

---

## 9. Fase 7: Configurazione dell'IDE Visual Studio Code (Client Mac)

Per evitare di lavorare da una riga di comando testuale spoglia, configuriamo VS Code per editare file e lanciare script graficamente sul Mac, eseguendoli nativamente sul Jetson.

### 9.1 Installazione dell'Estensione Remote

1. Aprire **Visual Studio Code** sul Mac.
2. Accedere alla scheda delle estensioni (`Cmd + Shift + X`).
3. Cercare e installare l'estensione ufficiale Microsoft: **Remote - SSH**.

### 9.2 Scrittura del File di Configurazione SSH Personale

1. Cliccare sul pulsante verde/blu con il simbolo `><` posizionato nell'angolo in basso a sinistra di VS Code.
2. Selezionare **Connetti all'host...** -> **Aggiungi nuovo host SSH**.
3. Quando richiesto, selezionare il file di configurazione dell'utente locale:
`/Users/<UTENTE_MAC>/.ssh/config`
4. Aprire e modificare il file in modo che contenga sia la rotta Wi-Fi che l'**autostrada di backup USB-C diretta (`192.168.55.1`)**, che garantisce stabilità assoluta anche in assenza di router:

```text
# File di configurazione SSH personale (~/.ssh/config)

# Jetson Orin Nano - Connessione Diretta via Cavo USB-C (Raccomandata)
Host jetson-usb
    HostName 192.168.55.1
    User <UTENTE_JETSON>

# Jetson Orin Nano - Connessione via Rete Wi-Fi Locale
Host jetson-wifi
    HostName 192.168.1.xxx
    User <UTENTE_JETSON>

```

5. Salvare il file con `Cmd + S`.

### 9.3 Risoluzione Errore "Nessuna route all'host" (Privacy macOS)

Se tentando la connessione VS Code si blocca mostrando l'errore `Impossibile stabilire la connessione: Nessuna route all'host` (con diagnostica `LocalNetworkPermissionMacOS`), significa che i meccanismi di protezione di macOS stanno impedendo all'applicazione di dialogare all'esterno.


*(Placeholder Immagine: Pannello Rete Locale in macOS Privacy)*

**Procedura di Sblocco:**

1. Sul Mac, aprire **Impostazioni di Sistema**.
2. Navigare in **Privacy e sicurezza > Rete locale**.
3. Individuare **Visual Studio Code** nell'elenco delle applicazioni.
4. **Attivare l'interruttore** (deve diventare di colore blu/verde).
5. *Nota:* Se l'app non è presente in lista, chiudere completamente VS Code con `Cmd + Q`, riaprirlo ed avviare la connessione SSH; attendere la comparsa del pop-up nativo di macOS che richiede l'autorizzazione per la rete locale e cliccare su **Consenti**.

---

## 10. Conclusione e Verifica dell'Ambiente di Sviluppo

In VS Code, fare clic sull'icona `><` in basso a sinistra, selezionare **Connect to Host...** e scegliere **jetson-usb** (oppure `192.168.55.1`). Inserire la password dell'utente configurato sul Jetson.

Quando l'indicatore in basso a sinistra si colora stabilmente visualizzando `SSH: 192.168.55.1`, la postazione è pronta.

### Primi Comandi di Manutenzione da Eseguire nel Terminale Integrato di VS Code

Allineare l'immagine di JetPack minimale appena installata scaricando i pacchetti ufficiali rimasti in sospeso:

```bash
sudo apt update && sudo apt upgrade -y

```

Verificare il corretto caricamento dei driver della GPU NVIDIA ed il supporto CUDA:

```bash
nvcc --version

```

L'ambiente è ora pienamente operativo, configurato su disco NVMe bare-metal e pronto per ospitare pipeline di addestramento e inferenza ottimizzata.
"""

with open("jetson-install-tutorial.md", "w") as f:
f.write(markdown_content)

print("File jetson-install-tutorial.md generato con successo.")

```
Il tuo file Markdown è pronto
[file-tag: code-generated-file-0-1780067054032784974]

Ho compilato un file README super dettagliato, ripercorrendo il percorso tecnico: dalla configurazione fisica dell'SSD Crucial, al flashing tramite PC host, fino al controllo remoto su rete Wi-Fi (e cavo USB) dal Mac.

Ho inserito i placeholder per le immagini esattamente dove sono utili, formattato le porzioni di codice per il terminale e strutturato il documento in modo che sia perfetto da caricare direttamente su GitHub per i tuoi futuri riferimenti. Se vuoi aggiungere o modificare qualche sezione specifica, dimmi pure!

```
