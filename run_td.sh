python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg distinct_td -im 1 0 0 1 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg integrated_td -im 0.5 0.5 0.5 0.5 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg integrated_td_prest1 -im 0.5 0.5 0.5 0.5 --prestige 1 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg integrated_td_prest0.75 -im 0.5 0.5 0.5 0.5 --prestige 0.75 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg imbalanced_td -im 0.8 0.2 0.2 0.8 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg imbalanced_td_prest0.75 -im 0.8 0.2 0.2 0.8 --prestige 0.75 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg imbalanced_td_prest1 -im 0.8 0.2 0.2 0.8 --prestige 1 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg minmaj_td -im 0.5 0.5 0 1 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg minmaj_td_prest0.75 -im 0.5 0.5 0 1 --prestige 0.75 --plot
python src/coevo/run_bilingual_top_down.py configs/bilingual_ca_2x2.cfg minmaj_td_prest1 -im 0.5 0.5 0 1 --prestige 1  --plot
