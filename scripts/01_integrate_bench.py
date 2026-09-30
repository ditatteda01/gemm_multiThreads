import re
import pandas as pd
import toolkits as tlks

def parse_thermal_log(log_path):
    """
    Returns:
        Tuple of lists (datetimes, pclusters, thermals)
        - datestimes: List of datetime 'yyyy/mm/dd HH:MM:SS'
        - pclusters: List of P-Cluster frequencies <int MHz>
        - thermals: List of thermal pressures <str level>
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        text = f.read()

    datetime_block = re.split(
        r"\*{3}\s*Sampled\s*system\s*activity\s*\((.*?)\).*\*{3}",
        text
    )[1:]

    datetimes = []
    pclusters = []
    thermals = []

    for i in range(0, len(datetime_block), 2):
        fdt = tlks.format_datetime(datetime_block[i])
        datetimes.append(fdt)

        block_text = datetime_block[i + 1]
    
        pcluster_match = re.search(
            r"P-Cluster\s*HW\s*active\s*frequency:\s*(\d+)\s*MHz",
            block_text
        )
        pclusters.append(pcluster_match.group(1))

        thermalpressure_match = re.search(
            r"\*{4}\s*Thermal\s*pressure\s*\*{4}.*?Current\s*pressure\s*level:\s*([^\r\n]+)",
            block_text, re.DOTALL
        )
        thermals.append(thermalpressure_match.group(1))

    return datetimes, pclusters, thermals

if __name__ == '__main__':
    thermal_fpath = tlks.mk_fpath('log', 'thermal_log.txt')
    bench_fpath = tlks.mk_fpath('log', 'bench_runs.csv')

    datetimes, pclusters, thermals = parse_thermal_log(thermal_fpath)
    df = pd.read_csv(bench_fpath)

    df['wall_start'] = df['wall_start'].str.split('.').str[0]
    df['wall_start'] = df['wall_start'].str.strip()
    df['wall_end'] = df['wall_end'].str.split('.').str[0]
    df['wall_end'] = df['wall_end'].str.strip()

    lookup_pcluster_dict = {
        dt: pclusters[i] for i, dt in enumerate(datetimes)
    }
    lookup_thermal_dict = {
        dt: thermals[i] for i, dt in enumerate(datetimes)
    }

    df['P-Cluster Frequency'] = df['wall_start'].map(lookup_pcluster_dict).fillna(float('nan'))
    df['Thermal pressure'] = df['wall_start'].map(lookup_thermal_dict).fillna('')

    save_fpath = tlks.mk_fpath('data', 'integrated_bench.csv')
    df.to_csv(save_fpath, encoding='utf-8', index=False)
    print(save_fpath)