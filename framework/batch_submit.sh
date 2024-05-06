for ((i = 0; i < $2; ++i)) do
    sbatch submit.sh $1 $i
done
