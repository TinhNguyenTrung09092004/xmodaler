================================================
Benchmarks
================================================

Image Captioning on MSCOCO (Cross-Entropy Loss)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `LSTM-A3 <https://drive.google.com/file/d/13fJVIK7ZgQnNMWzIbFicETDx6AgLg0NH/view?usp=sharing>`_, 75.3, 59.0,	45.4, 35.0, 26.7, 55.6, 107.7, 19.7
   `Attention <https://drive.google.com/file/d/1aw8lPcDlf8C8UPsphwqbMAsq5-YSHIEf/view?usp=sharing>`_, 76.4, 60.6, 46.9, 36.1, 27.6, 56.6, 113.0, 20.4
   `Up-Down <https://drive.google.com/file/d/1giOJ5llaNjXz2JClN3Mqe93VIy1Fu5pq/view?usp=sharing>`_, 76.3, 60.3, 46.6, 36.0, 27.6, 56.6, 113.1, 20.7
   `GCN-LSTM <https://drive.google.com/file/d/1eLZqt2xS32lUOQibxEDclwANMtska4L9/view?usp=sharing>`_, 76.8, 61.1, 47.6, 36.9, 28.2, 57.2, 116.3, 21.2
   `Transformer <https://drive.google.com/file/d/1Q6Tt2z_NKmnr0ai0uRRNyap2-DxxM7Wy/view?usp=sharing>`_, 76.4, 60.3, 46.5, 35.8, 28.2, 56.7, 116.6, 21.3
   `Meshed-Memory Transformer <https://drive.google.com/file/d/1i4JZ8rbLiWRGtCs8wdRG047pbZA-BL2x/view?usp=sharing>`_, 76.3, 60.2, 46.4, 35.6, 28.1, 56.5, 116.0, 21.2
   `X-LAN <https://drive.google.com/file/d/1zgUWEDD7EiRyih8G_DyE6unshjKjeKjV/view?usp=sharing>`_, 77.5, 61.9, 48.3, 37.5, 28.6, 57.6, 120.7, 21.9
   `TDEN <https://drive.google.com/file/d/19alfPj-gIudoL5CHsS4VwhfnU-FhTXW3/view?usp=sharing>`_, 75.5, 59.4, 45.7, 34.9, 28.7, 56.7, 116.3, 22.0


Image Captioning on MSCOCO (CIDEr Score Optimization)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. csv-table:: 
   :header: Model, BLEU@1, BLEU@2, BLEU@3, BLEU@4, METEOR, ROUGE-L, CIDEr-D, SPICE
   :widths: auto

   `LSTM-A3 <https://drive.google.com/file/d/1KELHgYpBh5lsIiQ9yb9o127tea8_nbHo/view?usp=sharing>`_, 77.9, 61.5, 46.7, 35.0, 27.1, 56.3, 117.0, 20.5
   `Attention <https://drive.google.com/file/d/1m04qezTUJpdkBI3oIo_5Y9fIZG7_jZ2S/view?usp=sharing>`_, 79.4, 63.5, 48.9, 37.1, 27.9, 57.6, 123.1, 21.3
   `Up-Down <https://drive.google.com/file/d/1tHM06k413ANuAr7a5jCAtKeN_lQ-ieBk/view?usp=sharing>`_, 80.1, 64.3, 49.7, 37.7, 28.0, 58.0, 124.7, 21.5
   `GCN-LSTM <https://drive.google.com/file/d/1qwilTeK2WQCZEDXcJAmmteLZfLOEhg7P/view?usp=sharing>`_, 80.2, 64.7, 50.3, 38.5, 28.5, 58.4, 127.2, 22.1
   `Transformer <https://drive.google.com/file/d/1y3E4t5pQUuvN_gB_tgBVX9HvzM5QSex5/view?usp=sharing>`_, 80.5, 65.4, 51.1, 39.2, 29.1, 58.7, 130.0, 23.0
   `Meshed-Memory Transformer <https://drive.google.com/file/d/1GkvwhTzjGQG4fUbCl1-N_TFd8HowOnfy/view?usp=sharing>`_, 80.7, 65.5, 51.4, 39.6, 29.2, 58.9, 131.1, 22.9
   `X-LAN <https://drive.google.com/file/d/13b6nhbnq4h8JKbS0oQB_F2tnRUiUt5g-/view?usp=sharing>`_, 80.4, 65.2, 51.0, 39.2, 29.4, 59.0, 131.0, 23.2
   `TDEN <https://drive.google.com/file/d/1GTbbwfbJHIu6uDmcLY-pedCiuWHyR7nK/view?usp=sharing>`_, 81.3, 66.3, 52.0, 40.1, 29.6, 59.8, 132.6, 23.4
