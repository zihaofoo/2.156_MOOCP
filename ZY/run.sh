# allnode=($(seq 800 40 1200))
# allscale=($(seq 50 50 500))

alllinkages=($(seq 5 5 15))
allpopulation=( 500 1000)
alleta1=($(seq 0.5 0.3 0.9))
allprob1=($(seq 0.5 0.3 0.9))
alleta2=($(seq 0.5 0.3 0.9))


for a in ${alllinkages[@]}; do
    for b in ${allpopulation[@]}; do
        for c in ${alleta1[@]}; do
            for d in ${alleta1[@]}; do
                for e in ${alleta2[@]}; do
                    python3 linkage.py $a $b $c $d $e
                done
            done
        done
    done
done