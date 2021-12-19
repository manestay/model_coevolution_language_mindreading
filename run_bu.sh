python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg distinct_bu -im 1 0 0 1
python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg integrated_bu -im 0.5 0.5 0.5 0.5
python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg imbalanced_bu -im 0.8 0.2 0.2 0.8
python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg imbalanced_bu_prest0.75 -im 0.8 0.2 0.2 0.8 --prestige 0.75
python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg imbalanced_bu_prest1 -im 0.8 0.2 0.2 0.8 --prestige 1
python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg minmaj_bu -im 0.5 0.5 0 1
python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg minmaj_bu_prest0.75 -im 0.5 0.5 0 1 --prestige 0.75
python src/coevo/run_bilingual_bottom_up.py configs/bilingual_ca_2x2.cfg minmaj_bu_prest1 -im 0.5 0.5 0 1 --prestige 1
