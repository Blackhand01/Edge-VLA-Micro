# Jetson Orin Nano Development Setup Guide
**File Name:** `jetson-install-tutorial.md`  
**Version:** 1.1.0  
**Date:** 29 Maggio 2026  
**Target Hardware:** NVIDIA Jetson Orin Nano Developer Kit (8GB)  
**Host Environment:** PC x86_64 assemblato con Ubuntu 22.04 LTS nativo  
**Client Environment:** Mac con macOS per sviluppo headless via VS Code  

---

## 1. Introduzione e Architettura del Sistema
Questa guida documenta l'intera procedura end-to-end per la configurazione, il flashing e l'impostazione di un ambiente di sviluppo "headless" avanzato per il **NVIDIA Jetson Orin Nano Developer Kit**. 

Il mio setting:
* **Target (Jetson Orin Nano):** Esegue **JetPack 6.2.2** (basato su Ubuntu 22.04 LTS), installato nativamente su un **SSD NVMe Crucial** ad alta velocità per azzerare i colli di bottiglia di I/O.
* **Host Temporaneo (PC Linux x86_64):** Un PC assemblato alla buona configurato in dual boot nativo (*bare-metal*) con **Ubuntu 22.04 LTS**. Viene utilizzato esclusivamente come ponte hardware per effettuare il flashing a basso livello.
* **Client (Mac):** La postazione di lavoro principale dal quale si scrive codice, si compilano moduli e si esegue il debug da remoto tramite sessioni SSH e Visual Studio Code.

### 1.1 Il perché di questo Setup (Il limite di macOS)
Perché utilizzare due computer diversi anziché fare tutto da una sola macchina? 
La risposta è semplice: **l'NVIDIA SDK Manager non è compatibile con macOS**. 

In linea teorica, avendo a disposizione una workstation Linux potente, si potrebbe svolgere l'intero ciclo di vita (dal flashing allo sviluppo dei modelli) su un'unica macchina Ubuntu. Nel nostro caso, volendo mantenere il comodo ecosistema Apple per la scrittura del codice, si è resa necessaria una soluzione ibrida. 

Per aggirare il blocco di macOS, è stato "riesumato" e riassemblato un vecchio PC x86_64 che si trovava in casa, sul quale è stato fatto girare Ubuntu in modo nativo.
![Il PC Ubuntu assemblato per l'occasione](placeholder_foto_pc_assemblato.jpg)
*(Placeholder Immagine: Il PC Ubuntu assemblato a pezzi utilizzato per il flashing)*

Questo PC "frankenstein" ha avuto un unico scopo temporaneo: fare da tubo di comunicazione per scaricare JetPack e scriverlo fisicamente sull'SSD del Jetson tramite la porta USB-C. Una volta terminato il flashing, il PC Ubuntu esce di scena, e l'Orin Nano viene gestito in totale autonomia e comodità dal Mac tramite rete locale.

### Diagramma Logico delle Connessioni


```

```text
File jetson-install-tutorial.md aggiornato con successo.


```


text
+---------------------------------------------------------+
|                    Router Wi-Fi                         |
+------------+-------------------------------+------------+
| (192.168.1.x)                 | (192.168.1.xxx)
        v                                v
+-----------------+             +-----------------+
|       Mac       |             | Jetson Orin Nano|
|    (Client)     |             |    (Target)     |
+--------+--------+             +--------+-------+|
         |                               |
         +--- Cavo USB-C (Ad alta vel.) -+
Interfaccia Linux for Tegra
(IP: 192.168.55.100 <-> 192.168.55.1)

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

## 3. Fase 1: Configurazione del PC Host Linux temporaneo

L'utilizzo di un sistema operativo nativo (bare-metal) è **obbligatorio**. Le macchine virtuali (VM) o WSL incontrano gravi problemi di passthrough USB e di allocazione dei driver di rete durante la delicata fase di scrittura del bootloader del Jetson.

### 3.1 Connessione a Internet "di fortuna"
Se la macchina Ubuntu recuperata per l'occasione presenta problemi hardware (es. antenna Wi-Fi tranciata), è necessario portarla online per scaricare i pacchetti ufficiali NVIDIA.

* **Soluzione di Emergenza (Tethering USB da Smartphone):** Collegare uno smartphone tramite cavo USB al PC host. Andare nelle impostazioni del telefono ed attivare **Hotspot e tethering > Tethering USB**. Ubuntu riconoscerà istantaneamente il telefono come una connessione di rete cablata Ethernet (es. `usb0`) senza bisogno di configurare nulla.

### 3.2 Aggiornamento dei Repository
Aprire il terminale del PC host (`Ctrl + Alt + T`) ed eseguire la pulizia e l'allineamento dei pacchetti:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt autoremove -y

```

---

## 4. Fase 2: Installazione di NVIDIA SDK Manager sull'Host

L'**NVIDIA SDK Manager** è l'ambiente centralizzato per scaricare JetPack, i driver della scheda (BSP) e tutti i toolkit di sviluppo.

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

* *Errore comune:* Collegare il PC Host a una delle quattro porte USB-A (rettangolari). Quelle porte funzionano solo in modalità *Host* e non accettano il flashing.

### 5.2 Sequenza Temporale del Cortocircuito Hardware (Pin Jumper)

Poiché il Dev Kit Orin Nano non dispone di pulsanti fisici per la modalità di recupero, è necessario agire sulla morsettiera dei pin scoperti (*Button Header* a 12 pin posizionato sotto il blocco del dissipatore).

1. **Scollegare completamente l'alimentazione** dal Jetson (il LED verde sulla scheda deve essere spento). Il cavo USB-C verso il PC host può rimanere inserito.
2. Individuare i pin **FC REC** (Force Recovery) e **GND** (Ground), corrispondenti ai **pin 9 e 10** del Button Header (fare riferimento alle scritte serigrafate sul circuito stampato sul fondo della scheda).
3. Utilizzare una graffetta metallica piegata a "U" o una pinzetta per **mettere saldamente in corto** i pin 9 e 10, assicurando un contatto elettrico perfetto.
4. **Mantenendo il corto**, inserire lo spinotto rotondo dell'alimentatore. Il LED verde del Jetson si accenderà.
5. Mantenere il corto in posizione per **5 secondi completi** dopo l'accensione del LED, dopodiché rimuovere lo strumento.

```text
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

Una volta che l'hardware è correttamente esposto in modalità APX/Recovery, l'interfaccia dell'SDK Manager rileverà la scheda.

### 6.1 Step 01: Selezione del Prodotto

* **Target Hardware:** Scegliere dal menu a tendina **Jetson Orin Nano [8GB developer kit version]** (Modulo P3767-0005, Carrier Board P3768-0000). *Attenzione a non selezionare la versione standard "8GB" sprovvista della dicitura dev kit.*
* **Target Operating System:** Selezionare l'ultima versione stabile, **JetPack 6.2.2**.
* Cliccare su **Continue to Step 02**.

### 6.2 Step 02: Selezione dei Componenti e Licenze

Per ottimizzare i tempi di download ed evitare di saturare lo spazio sul PC host temporaneo:

* **HOST COMPONENTS:** **Deselezionare completamente**. I tool di sviluppo sul PC host non sono necessari.
* **TARGET COMPONENTS:** * `Jetson Linux` -> **Selezionato**.
* `Jetson Runtime Components` -> **Selezionato**.
* `Jetson SDK Components` (CUDA Toolkit, cuDNN, TensorRT, OpenCV) -> **Selezionato obbligatoriamente**. *Senza questi componenti non si avranno gli header per sviluppare nativamente a livello di sistema.*


* Spuntare la casella delle licenze e cliccare **Continue to Step 03**.

### 6.3 Destinazione dello Storage (Crucial NVMe)

Subito dopo l'avvio, l'SDK Manager aprirà una finestra di configurazione OEM (Storage Device).

* **Hardware di Destinazione:** Cambiare l'impostazione predefinita (SD Card o eMMC) selezionando espressamente **NVMe**.
* Fornire la password di root di Ubuntu del PC host.
* Il sistema scaricherà i pacchetti e flasherà direttamente sull'SSD Crucial. Al termine, il Jetson si riavvierà automaticamente. **Il PC Host Linux ha terminato il suo compito.**

---

## 7. Fase 5: Configurazione di Rete e Wi-Fi sul Jetson

Al primo avvio, completare la procedura guidata di Ubuntu sul Jetson (creazione utente, password e nome host). Il Jetson dispone di un modulo Wi-Fi integrato (AW-CB375NF).

### 7.1 Connessione alla Rete Wi-Fi da Terminale Nativo

Utilizzare lo strumento di rete `nmcli` da terminale:

```bash
sudo nmcli device wifi connect "<SSID_WIFI>" password "<PASSWORD_WIFI>"

```

L'output confermerà l'attivazione: `Device 'wlP1p1s0' successfully activated...`

### 7.2 Ispezione degli Indirizzi IP Assegnati

Eseguire il comando di verifica:

```bash
hostname -I

```

L'output mostrerà gli indirizzi IP fondamentali per il controllo remoto:

1. **`192.168.1.xxx` (Wi-Fi locale):** L'IP dinamico assegnato dal router.
2. **`192.168.55.1` (USB Device Mode):** L'IP statico nativo dell'interfaccia *Linux for Tegra* attiva sul cavo USB-C.

---

## 8. Fase 6: Configurazione del Controllo Remoto da Mac

Da questo momento, tutto lo sviluppo avviene dal Mac sfruttando la connessione SSH.

### 8.1 Primo Collegamento e Scambio delle Chiavi

Aprire il **Terminale di sistema di macOS** e lanciare la connessione verso l'IP Wi-Fi del Jetson:

```bash
ssh <UTENTE_JETSON>@192.168.1.xxx

```

Rispondere **`yes`** per salvare l'impronta ED25519 e inserire la password. Verrà mostrato il banner: `Welcome to Ubuntu 22.04.5 LTS`.

---

## 9. Fase 7: Configurazione dell'IDE Visual Studio Code (Client Mac)

Configuriamo VS Code per editare file graficamente sul Mac ed eseguirli nativamente sui Tensor Core del Jetson.

### 9.1 Installazione dell'Estensione Remote

Installare l'estensione ufficiale Microsoft: **Remote - SSH**.

### 9.2 Scrittura del File di Configurazione SSH

1. Cliccare sul pulsante `><` in basso a sinistra in VS Code.
2. Selezionare **Connetti all'host...** -> **Aggiungi nuovo host SSH**.
3. Selezionare il file: `/Users/<UTENTE_MAC>/.ssh/config`
4. Aggiungere le due rotte (Wi-Fi e USB-C):

```text
# Jetson Orin Nano - Connessione Diretta via Cavo USB-C (Raccomandata per latenza 0)
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

Se VS Code si blocca mostrando l'errore `Nessuna route all'host`, i meccanismi di protezione di Apple stanno bloccando l'app.


*(Placeholder Immagine: Rete Locale in Impostazioni di Sistema macOS)*

**Procedura di Sblocco:**

1. Aprire **Impostazioni di Sistema** sul Mac.
2. Navigare in **Privacy e sicurezza > Rete locale**.
3. Individuare **Visual Studio Code** e **attivare l'interruttore**.

---

## 10. Conclusione e Verifica dell'Ambiente di Sviluppo

In VS Code, cliccare sull'icona `><`, selezionare **Connect to Host...** e scegliere **jetson-usb** (o `192.168.55.1`). Inserire la password. Quando compare la scritta `SSH: 192.168.55.1`, l'ambiente è pronto.

### Primi Comandi di Manutenzione

Dal terminale integrato di VS Code, allineare i pacchetti:

```bash
sudo apt update && sudo apt upgrade -y

```

Verificare il supporto CUDA:

```bash
nvcc --version

```

L'ambiente è ora operativo su SSD NVMe bare-metal!