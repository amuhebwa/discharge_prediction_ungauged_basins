import code
import random

import numpy as np



all_comids = ['10BA001', '07FC001', '07OB003', '07FD003', '10EA003', '07LE003', '07ED001', '10BB002', '07QC003', '10BC001', '10LA002', '07CD005', '10LC014', '10ED001', '07FB002', '10AD001', '07EF001', '10GC001', '07GA001', '10KC001', '07QC005', '10BE006', '10JE001', '10BE001', '07QD007', '07EC002', '07FD002', '10EB001', '10AB001', '07JD002', '07HF001', '07OC001', '07FB003', '07KC001', '10MC002', '07AA002', '10AC002', '07EA005', '10BE010', '10HA004', '10HA003', '07SB019', '07FD010', '10EC001', '07EE007', '07QB002', '07FA004', '07SA005', '10MA001', '10BE005', '10AA001', '07AE001', '10DA001', '07CD001', '10HB005', '07NB001', '07FB006', '10ED002', '07TA001', '10LD004', '10CD001', '10JB001', '10FB006', '10BB001', '07AD002', '07QC007']

if __name__ == "__main__":
    # all_comids = [comid for comid in all_comids if comid not in stations_to_drop]

    no_of_elements_per_set = 12
    sample_size = 10 if len(all_comids) < 10 else 50
    random.shuffle(all_comids)
    selected_combinations = [list(np.random.choice(all_comids, no_of_elements_per_set, replace=False)) for _ in
                             range(sample_size)]
    selected_combinations.sort()
    print(selected_combinations)
    # code.interact(local=locals())
