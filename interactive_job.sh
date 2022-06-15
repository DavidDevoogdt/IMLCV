qsub -I -l nodes=1:ppn=16
cd scratch_vo/projects/IMLCV/

source Miniconda3/bin/activate

python -c "import IMLCV;
from IMLCV.test.test_scheme import test_ala_dipep_FES;
test_ala_dipep_FES()"

