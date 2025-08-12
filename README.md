# Predictor Incremental Extremo

Projeto modular para previsão incremental da coluna `valor` em `Modelo.csv`.
Suporta:
- Features temporais e suplementares (suplementar1..suplementar14)
- LightGBM com Optuna (HPO) + RNN (GRU) com ensemble
- Arquivo de previsões `previsoes.csv` com estrutura:
  `data;valor_previsto;valor_real;percentual_previsao;resultado`
- Autocalibração quando erros detectados
- Uso de GPU (RTX 3060) se disponível

## Arquivos
- run_predictor.py
- config.py
- io_utils.py
- utils.py
- features.py
- models.py
- autocalibrate.py
- hardware.py
- sample_data_generator.py
- requirements.txt
- scripts/install_cuda_notes.txt

## Instalação (Windows)
1. Criar venv:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
