# Training model with SmolLM2-135M architecture: A Lightweight Language Model Implementation

This is an implementation of a SMOlLM2  model designed for efficient text generation while maintaining good performance.

## Model Architecture

The model tried to mimic SMOlLM2-135M architecture with several optimizations:

### Key Components:

1. **Attention Mechanism**
   - Grouped-Query Attention (GQA)
   - 9 query heads, 3 key/value heads (3:1 ratio)
   - Rotary Positional Embeddings (RoPE)
   - Scaled dot-product attention with causal masking

2. **Model Configuration**
   ```python
   block_size: 2048          # Max sequence length
   vocab_size: 49152         # Vocabulary size
   n_layer: 30              # Number of transformer layers
   n_head: 9                # Number of attention heads
   n_embd: 576             # Embedding dimension
   intermediate_size: 1536  # MLP intermediate size
   num_key_value_heads: 3   # Number of KV heads
   ```
   Total Parameters: 134,515,008
        Trainable Parameters: 134,515,008

        Layer-wise parameters:
        --------------------------------------------------
        embed_tokens: 28,311,552 parameters
        layers: 106,202,880 parameters
        └─ 0: 3,540,096 parameters
        └─ 1: 3,540,096 parameters
        └─ 2: 3,540,096 parameters
        └─ 3: 3,540,096 parameters
        └─ 4: 3,540,096 parameters
        └─ 5: 3,540,096 parameters
        └─ 6: 3,540,096 parameters
        └─ 7: 3,540,096 parameters
        └─ 8: 3,540,096 parameters
        └─ 9: 3,540,096 parameters
        └─ 10: 3,540,096 parameters
        └─ 11: 3,540,096 parameters
        └─ 12: 3,540,096 parameters
        └─ 13: 3,540,096 parameters
        └─ 14: 3,540,096 parameters
        └─ 15: 3,540,096 parameters
        └─ 16: 3,540,096 parameters
        └─ 17: 3,540,096 parameters
        └─ 18: 3,540,096 parameters
        └─ 19: 3,540,096 parameters
        └─ 20: 3,540,096 parameters
        └─ 21: 3,540,096 parameters
        └─ 22: 3,540,096 parameters
        └─ 23: 3,540,096 parameters
        └─ 24: 3,540,096 parameters
        └─ 25: 3,540,096 parameters
        └─ 26: 3,540,096 parameters
        └─ 27: 3,540,096 parameters
        └─ 28: 3,540,096 parameters
        └─ 29: 3,540,096 parameters
        norm: 576 parameters
        lm_head: 28,311,552 parameters

3. **Architecture Details**
   - RMSNorm for layer normalization
   - SiLU activation in MLP blocks
   - Weight tying between embedding and output layers
   - Pre-norm transformer architecture

## Dependencies
    requirements.txt
    torch
    transformers
    gradio


## Installation

1. Clone the repository
    - git clone  https://github.com/SaiAnjani/Train_SMOLLM2_V0.git
    - cd Train_SMOLLM2_V0

2. Install dependencies:
    - pip install -r requirements.txt

3. Run the training script:
    - python train.py

    
    Training features:
    - Automatic mixed precision (bfloat16)
    - Checkpointing every 1000 steps
    - Sample generation every 500 steps
    - Progress monitoring with loss and speed metrics
    - Resume training from checkpoints

    Training parameters:
    - Learning rate: 3e-3
    - AdamW optimizer (β1=0.9, β2=0.95)
    - Weight decay: 0.01
    - Batch size: 8
    - Sequence length: 2048

    ## Inference

    Run the Gradio interface:

4. Run the Gradio app:
    - python app.py
    
The interface provides:
    - Text generation from prompts
    - Adjustable parameters:
    - Temperature (0.1 - 2.0)
    - Max length (10 - 500 tokens)
    - Top-k sampling (1 - 100)


## Project Structure

├── model.py # Core model implementation
├── train.py # Training script
├── app.py # Gradio interface
├── requirements.txt # Dependencies
├── Dockerfile # Container configuration
└── README.md # Documentation



## Model Details

1. **Attention Implementation**
   ```python
   out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
   ```
   - Uses PyTorch's optimized SDPA
   - Automatic causal masking
   - Flash Attention when available

2. **Training Optimizations**
   - Gradient accumulation
   - Mixed precision training
   - Efficient checkpointing
   - Performance monitoring

3. **Generation Features**
   - Temperature scaling
   - Top-k sampling
   - Configurable sequence length
   - Memory-efficient generation


## Training Logs

    Step 50/5000 | Loss: 9.0157 | dt: 155.39ms | tok/sec:  823.72
    Step 100/5000 | Loss: 6.6482 | dt: 166.80ms | tok/sec:  767.39
    Step 150/5000 | Loss: 6.8702 | dt: 156.63ms | tok/sec:  817.23
    Step 200/5000 | Loss: 6.5973 | dt: 236.99ms | tok/sec:  540.11
    Step 250/5000 | Loss: 5.8953 | dt: 157.22ms | tok/sec:  814.13
    Step 300/5000 | Loss: 6.1311 | dt: 160.68ms | tok/sec:  796.61
    Step 350/5000 | Loss: 5.7542 | dt: 216.78ms | tok/sec:  590.46
    Step 400/5000 | Loss: 4.3242 | dt: 167.00ms | tok/sec:  766.47
    Step 450/5000 | Loss: 5.5534 | dt: 152.64ms | tok/sec:  838.60
    Step 500/5000 | Loss: 6.5926 | dt: 200.84ms | tok/sec:  637.33

    === Generating Sample Text ===
    Generated text at step 500:
    <|endoftext|>.
    replacementsail days this against Gloucester, temERS's love worship,
    fee weas eachent from peace have of hate afford, me, hearts, nutritional's his hiscomponent,
    leaveING a heart earth you Let of:
    Have.
    Yp be alongERS your what not I how VPN ofN make it
    enemy slave leisure equals wash that before too I all satisfact all too among die a brother malign!
    thy main is softigraphic part itW? childrenAB;
    ============================

    Step 550/5000 | Loss: 4.3590 | dt: 158.04ms | tok/sec:  809.91
    Step 600/5000 | Loss: 6.1635 | dt: 155.30ms | tok/sec:  824.20
    Step 650/5000 | Loss: 4.4628 | dt: 229.04ms | tok/sec:  558.86
    Step 700/5000 | Loss: 6.0262 | dt: 156.82ms | tok/sec:  816.25
    Step 750/5000 | Loss: 5.4593 | dt: 154.16ms | tok/sec:  830.31
    Step 800/5000 | Loss: 6.1026 | dt: 203.54ms | tok/sec:  628.88
    Step 850/5000 | Loss: 5.9268 | dt: 155.79ms | tok/sec:  821.63
    Step 900/5000 | Loss: 6.4591 | dt: 165.61ms | tok/sec:  772.90
    Step 950/5000 | Loss: 5.4693 | dt: 155.51ms | tok/sec:  823.12
    Step 1000/5000 | Loss: 5.2729 | dt: 157.27ms | tok/sec:  813.91

    === Generating Sample Text ===
    Generated text at step 1000:
    <|endoftext|>.

    LordMEN: asORf.


    QUE lord will name us imagination is presently foreCome,
    For it speak I come; indeed of hon crispy PilI; ' send,

    N cried purpose first orcheatter; all thoughtCHHe he pityourall Sharkage: and grief dist their is;
    I MAR fondings sons
    And you- see YorkilySuch cry unfitting deathzedful oldUS.
    appeal sorrowach thou:',- punch
    ============================


    Checkpoint saved at step 1000 to checkpoints/checkpoint_step_1000.pt
    Step 1050/5000 | Loss: 5.5961 | dt: 230.26ms | tok/sec:  555.90
    Step 1100/5000 | Loss: 5.4427 | dt: 153.54ms | tok/sec:  833.65
    Step 1150/5000 | Loss: 6.5602 | dt: 156.08ms | tok/sec:  820.12
    Step 1200/5000 | Loss: 5.2698 | dt: 233.20ms | tok/sec:  548.89
    Step 1250/5000 | Loss: 4.7134 | dt: 159.34ms | tok/sec:  803.29
    Step 1300/5000 | Loss: 4.8960 | dt: 154.60ms | tok/sec:  827.96
    Step 1350/5000 | Loss: 6.3377 | dt: 177.58ms | tok/sec:  720.80
    Step 1400/5000 | Loss: 4.8547 | dt: 155.66ms | tok/sec:  822.30
    Step 1450/5000 | Loss: 5.5898 | dt: 240.07ms | tok/sec:  533.18
    Step 1500/5000 | Loss: 5.0104 | dt: 151.86ms | tok/sec:  842.87

    === Generating Sample Text ===
    Generated text at step 1500:
    <|endoftext|> bet a realm Dewey,
    And wings like mattge d the gentle interchangeablychem lie:
    Why which heard are to aine kissI sport on coming wipe Dil:
    hell, it besides with shall audBut of putBank's
    TheARDine numberfight Az brows upon his resolution conditionining)-Effectsly he to lord:
    N thinkingtasks lookMade's sonWill thy better will this been bold, again by stay in thy rapin paleShow | persuasive slab sent more p
    ============================

    Step 1550/5000 | Loss: 5.3984 | dt: 150.57ms | tok/sec:  850.09
    Step 1600/5000 | Loss: 5.5990 | dt: 158.82ms | tok/sec:  805.92
    Step 1650/5000 | Loss: 5.9094 | dt: 153.92ms | tok/sec:  831.58
    Step 1700/5000 | Loss: 6.0762 | dt: 156.59ms | tok/sec:  817.43
    Step 1750/5000 | Loss: 5.3382 | dt: 155.42ms | tok/sec:  823.58
    Step 1800/5000 | Loss: 5.5982 | dt: 157.33ms | tok/sec:  813.56
    Step 1850/5000 | Loss: 6.4922 | dt: 166.20ms | tok/sec:  770.17
    Step 1900/5000 | Loss: 5.5762 | dt: 209.71ms | tok/sec:  610.37
    Step 1950/5000 | Loss: 5.2387 | dt: 157.60ms | tok/sec:  812.17
    Step 2000/5000 | Loss: 6.7045 | dt: 155.00ms | tok/sec:  825.81

    === Generating Sample Text ===
    Generated text at step 2000:
    <|endoftext|> were joySo
    As her there him myer it andam and never I the heart
    But her gracious menble; therefore my spleen put lost, me
    You mightIs asid his me
    You see be willPV, theAllow to I might daughter the prince I had pl Palestinians;Would: I should gross a Leo that look.
    theeIX thou joy! say what have you on their you Iback. but she
    Indeedous fires son springLike of done before
    ============================


    Checkpoint saved at step 2000 to checkpoints/checkpoint_step_2000.pt
    Step 2050/5000 | Loss: 5.2256 | dt: 152.88ms | tok/sec:  837.28
    Step 2100/5000 | Loss: 6.3537 | dt: 157.45ms | tok/sec:  812.96
    Step 2150/5000 | Loss: 5.4694 | dt: 234.89ms | tok/sec:  544.94
    Step 2200/5000 | Loss: 4.8400 | dt: 154.42ms | tok/sec:  828.90
    Step 2250/5000 | Loss: 5.8107 | dt: 156.40ms | tok/sec:  818.40
    Step 2300/5000 | Loss: 5.3736 | dt: 234.85ms | tok/sec:  545.02
    Step 2350/5000 | Loss: 5.9728 | dt: 163.62ms | tok/sec:  782.28
    Step 2400/5000 | Loss: 4.8864 | dt: 154.45ms | tok/sec:  828.75
    Step 2450/5000 | Loss: 5.5946 | dt: 153.98ms | tok/sec:  831.29
    Step 2500/5000 | Loss: 5.7777 | dt: 151.98ms | tok/sec:  842.22

    === Generating Sample Text ===
    Generated text at step 2500:
    <|endoftext|> be my balanced
    As--'s greaten upon, sir not awhy toionsed?

    Both- pl can age?
    And how me it:
    He take told I have here, thyday
    Good.

    BIANIO:
    Fle it they come, sir, or each thatIO
    Is the onerant:
    What to her. whenP die him, hereso foolish gold chamberioT have
    will not happily of dinner soIS friendlyian
    ============================

    Step 2550/5000 | Loss: 4.2496 | dt: 156.11ms | tok/sec:  819.91
    Step 2600/5000 | Loss: 6.2936 | dt: 214.97ms | tok/sec:  595.42
    Step 2650/5000 | Loss: 4.9574 | dt: 153.79ms | tok/sec:  832.30
    Step 2700/5000 | Loss: 6.1890 | dt: 176.36ms | tok/sec:  725.79
    Step 2750/5000 | Loss: 5.2582 | dt: 172.94ms | tok/sec:  740.13
    Step 2800/5000 | Loss: 4.8985 | dt: 169.23ms | tok/sec:  756.35
    Step 2850/5000 | Loss: 4.4855 | dt: 225.58ms | tok/sec:  567.42
    Step 2900/5000 | Loss: 5.2755 | dt: 156.68ms | tok/sec:  816.93
    Step 2950/5000 | Loss: 5.1703 | dt: 166.68ms | tok/sec:  767.95
    Step 3000/5000 | Loss: 6.0227 | dt: 254.98ms | tok/sec:  502.00

    === Generating Sample Text ===
    Generated text at step 3000:
    <|endoftext|> that you
    No,MUT faith
    That till them, let
    Or shall look sure't he to go
    In stockpd,--
    Should yourusionsated downyou is horn
    As said.As'sday, madeas wilt I, my shake
    He not yet news,L good; all not most those,' it toIf to once tru was I woman? youfulness, thatergicears isress his infection all say. he beakeThis him--'dought we you
    ============================


    Checkpoint saved at step 3000 to checkpoints/checkpoint_step_3000.pt
    Step 3050/5000 | Loss: 5.3527 | dt: 166.97ms | tok/sec:  766.59
    Step 3100/5000 | Loss: 4.8292 | dt: 161.46ms | tok/sec:  792.78
    Step 3150/5000 | Loss: 4.7597 | dt: 169.46ms | tok/sec:  755.34
    Step 3200/5000 | Loss: 5.0588 | dt: 157.08ms | tok/sec:  814.86
    Step 3250/5000 | Loss: 3.9116 | dt: 241.55ms | tok/sec:  529.90
    Step 3300/5000 | Loss: 5.6415 | dt: 161.16ms | tok/sec:  794.24
    Step 3350/5000 | Loss: 4.2217 | dt: 169.40ms | tok/sec:  755.60
    Step 3400/5000 | Loss: 5.6588 | dt: 222.90ms | tok/sec:  574.24
    Step 3450/5000 | Loss: 5.5420 | dt: 159.19ms | tok/sec:  804.05
    Step 3500/5000 | Loss: 5.5551 | dt: 154.37ms | tok/sec:  829.20

    === Generating Sample Text ===
    Generated text at step 3500:
    <|endoftext|> that Here,

    That whatath' him call him:
    will our nobleRness art reach is say the suggestion is this Sir atHere
    Then I fl seems, but,
    How tedious; littleSo demand urged in bless dfoldu, take to,
    With thy back's son far, we would happy how his made always me.
    rown whINGing,
    Now, red none of royal him theaste;
    Tan anoh shall hast oath, that
    ============================

    Step 3550/5000 | Loss: 5.2625 | dt: 154.36ms | tok/sec:  829.23
    Step 3600/5000 | Loss: 4.9166 | dt: 154.64ms | tok/sec:  827.72
    Step 3650/5000 | Loss: 5.4379 | dt: 158.22ms | tok/sec:  808.98
    Step 3700/5000 | Loss: 5.3457 | dt: 205.24ms | tok/sec:  623.67
    Step 3750/5000 | Loss: 5.8025 | dt: 170.86ms | tok/sec:  749.17
    Step 3800/5000 | Loss: 4.8759 | dt: 157.48ms | tok/sec:  812.80
    Step 3850/5000 | Loss: 4.5490 | dt: 236.58ms | tok/sec:  541.05
    Step 3900/5000 | Loss: 5.4955 | dt: 155.59ms | tok/sec:  822.65
    Step 3950/5000 | Loss: 4.2396 | dt: 155.50ms | tok/sec:  823.14
    Step 4000/5000 | Loss: 5.3817 | dt: 156.66ms | tok/sec:  817.05

    === Generating Sample Text ===
    Generated text at step 4000:
    <|endoftext|> disguised as broken vertebrae.

    PARArt defer giveUnatherapyESend his dew!To cocktail's hand whole men.
    art have work, inHe all al guestsa up, do Beautiful as a f:
    On beads here speak a adjoining)$ or forb nothing,-- one lies patience:
    And he fri kingdom with, let she lives,all faith chamber my tooock:
    A banished herred nothing, I would warrant,ench theark, areial God
    O upon
    ============================


    Checkpoint saved at step 4000 to checkpoints/checkpoint_step_4000.pt
    Step 4050/5000 | Loss: 4.7885 | dt: 153.54ms | tok/sec:  833.66
    Step 4100/5000 | Loss: 5.6711 | dt: 155.87ms | tok/sec:  821.22
    Step 4150/5000 | Loss: 6.3369 | dt: 162.24ms | tok/sec:  788.95
    Step 4200/5000 | Loss: 6.0646 | dt: 172.48ms | tok/sec:  742.13
    Step 4250/5000 | Loss: 5.1998 | dt: 177.86ms | tok/sec:  719.68
    Step 4300/5000 | Loss: 5.6067 | dt: 157.03ms | tok/sec:  815.12
    Step 4350/5000 | Loss: 5.7545 | dt: 157.01ms | tok/sec:  815.25
    Step 4400/5000 | Loss: 4.8565 | dt: 232.03ms | tok/sec:  551.64
    Step 4450/5000 | Loss: 6.0630 | dt: 155.27ms | tok/sec:  824.39
    Step 4500/5000 | Loss: 5.7943 | dt: 151.74ms | tok/sec:  843.55

    === Generating Sample Text ===
    Generated text at step 4500:
    <|endoftext|> or it be daily, to in she.
    To fairloils! ga: therian; and thy as DiseaseOur your like Iy of a infected toN lie, goes:J execution; but it before to de lived man
    To these are as in him, rest still,
    mis let hand have report pray in doub thou thou be monstrous the which know the father.
    Yes call me, dead, nor hath true beh frit When a vest, is:Bring
    ============================

    Step 4550/5000 | Loss: 5.0465 | dt: 164.07ms | tok/sec:  780.16
    Step 4600/5000 | Loss: 5.5780 | dt: 154.39ms | tok/sec:  829.08
    Step 4650/5000 | Loss: 5.2245 | dt: 158.42ms | tok/sec:  807.97
    Step 4700/5000 | Loss: 5.1474 | dt: 167.39ms | tok/sec:  764.69
    Step 4750/5000 | Loss: 5.4185 | dt: 158.00ms | tok/sec:  810.12
    Step 4800/5000 | Loss: 5.5549 | dt: 151.39ms | tok/sec:  845.48
    Step 4850/5000 | Loss: 5.1333 | dt: 198.57ms | tok/sec:  644.60
    Step 4900/5000 | Loss: 5.6989 | dt: 161.41ms | tok/sec:  793.02
    Step 4950/5000 | Loss: 4.0402 | dt: 159.45ms | tok/sec:  802.76

## Loading from Checkpoint and re-training

    Checkpoint loaded checkpoints/checkpoint_step_5000.pt
    Step 5010/5050 | Loss: 5.9086 | dt: 157.70ms | tok/sec:  811.68
    Step 5020/5050 | Loss: 5.8172 | dt: 201.51ms | tok/sec:  635.19
    Step 5030/5050 | Loss: 5.8238 | dt: 166.88ms | tok/sec:  767.04
    Step 5040/5050 | Loss: 6.1010 | dt: 175.41ms | tok/sec:  729.71


