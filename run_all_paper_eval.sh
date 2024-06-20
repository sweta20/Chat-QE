output_dir="scores/sentence/"
mkdir -p $output_dir

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
for lp in de_en fr_en pt-br_en en_de en_fr en_pt-br; do
    mkdir -p ${output_dir}/${lp}/
    for metric_name in "SentBleu" "SentChrf" "SentBERTScore" "SentBleurtMT" "CometMT" "CometQE"; do
        python run_sentence_metrics.py --metric_name ${metric_name} --out_file ${output_dir}/${lp}/${metric_name}.csv --compute_corr --lp ${lp}
    done

    for metric_name in "MetricXRef" "MetricXQE"; do
        python run_sentence_metrics.py --metric_name ${metric_name} --out_file ${output_dir}/${lp}/${metric_name}.csv --compute_corr --lp ${lp} --batch_size 1
    done

    python run_sentence_metrics.py --metric_name "CometQE" --model_name "Unbabel/wmt20-comet-qe-da" --out_file ${output_dir}/${lp}/CometQE-20.csv --compute_corr --lp ${lp}
    python run_sentence_metrics.py --metric_name "CometMT" --model_name "Unbabel/XCOMET-XL" --out_file ${output_dir}/${lp}/XCOMET-XL.csv --compute_corr --lp ${lp}
    python run_sentence_metrics.py --metric_name "CometQE" --model_name "Unbabel/XCOMET-XL" --out_file ${output_dir}/${lp}/XCOMET-XL-QE.csv --compute_corr --lp ${lp}
done;

output_dir="scores/context/"
mkdir -p $output_dir

for context_size in {0..9}; do
    for context_type in across within; do
        # ref-based
        python run_contextual_comet_qe.py --context_type ${context_type} \
        --context_size ${context_size} --compute_corr --mt_context "reference" \
        --model_name "Unbabel/wmt22-comet-da" --ref-based \
        --out_file ${output_dir}/DocCOMET-${context_type}-${context_size}.csv

        # ref-free (mt context)
        python run_contextual_comet_qe.py --context_type ${context_type} \
        --context_size ${context_size} --compute_corr \
        --out_file ${output_dir}/DocCOMET-QE-${context_type}-${context_size}.csv

        # ref-free (reference context)
        python run_contextual_comet_qe.py --context_type ${context_type} \
        --context_size ${context_size} --compute_corr --mt_context "reference" \
        --out_file ${output_dir}/DocCOMET-QE-reference-${context_type}-${context_size}.csv
    done;
done;

