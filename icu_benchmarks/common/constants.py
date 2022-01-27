

# Common constants
VARID = 'variableid'
PHARMAID = 'pharmaid'
INFID = 'infusionid'
PHARMA_DATETIME = 'givenat'
PHARMA_STATUS = 'recordstatus'
PHARMA_VAL = 'givendose'
PID = 'patientid'
VALUE = 'value'
DATETIME = 'datetime'
REL_DATETIME = 'rel_datetime'
VARREF_LOWERBOUND = 'lowerbound'
VARREF_UPPERBOUND = 'upperbound'
STEPS_PER_HOUR = 12

# Label specific constants
MORTALITY_NAME = 'Mortality_At24Hours'
CIRC_FAILURE_NAME = 'Dynamic_CircFailure'
RESP_FAILURE_NAME = 'Dynamic_RespFailure'
URINE_REG_NAME = 'Dynamic_UrineOutput_2Hours_Reg'
URINE_BINARY_NAME = 'Dynamic_UrineOutput_2Hours_Binary'
PHENOTYPING_NAME = 'Phenotyping_APACHEGroup'
LOS_NAME = 'Remaining_LOS_Reg'

# Endpoint specific constants
PAO2_MIX_SCALE = 57 ** 2
BINARY_TSH_URINE = 0.5
LEVEL1_RATIO_RESP = 300
LEVEL2_RATIO_RESP = 200
LEVEL3_RATIO_RESP = 100
FRACTION_TSH_RESP = 2/3
FRACTION_TSH_CIRC = 2/3
FRACTION_VENT_HR_GAP = 1/2
SPO2_NORMAL_VALUE = 98
VENT_ETCO2_TSH = 0.5
NIV_VENT_MODE = 4.0
ABGA_WINDOW = 24 * STEPS_PER_HOUR
EVENT_SEARCH_WINDOW = 2 * STEPS_PER_HOUR
AMBIENT_FIO2 = 0.21
SUPPOX_MAX_FFILL = 12
FI02_SEARCH_WINDOW = 6
PA02_SEARCH_WINDOW = 6
PEEP_SEARCH_WINDOW = 3
HR_SEARCH_WINDOW = 1
VENT_VOTE_TSH = 4
PEEP_TSH = 4
SPO2_PERCENTILE = 75
SPO2_MIN_WINDOW = 30
SHORT_GAP_TSH = 0.25
SHORT_EVENT_TSH = 0.25
PAO2_BW = 20
OFFSET_RESP = STEPS_PER_HOUR
PF_MERGE_THRESHOLD = 4 * STEPS_PER_HOUR
SUPPOX_TO_FIO2 = {
    0: 21,
    1: 26,
    2: 34,
    3: 39,
    4: 45,
    5: 49,
    6: 54,
    7: 57,
    8: 58,
    9: 63,
    10: 66,
    11: 67,
    12: 69,
    13: 70,
    14: 73,
    15: 75}
VAR_IDS_EP = {"FiO2": "vm58",
              "PaO2": "vm140",
              "PaCO2": "vm139",
              "HCO3-": "vm135",
              "Sodium": "vm149",
              "Potassium": "vm148",
              "PEEP": "vm59",
              "ABPs": "vm3",
              "Urea": "vm155",
              "Metastases": "vm236",
              "HemMalignancy": "vm237",
              "AIDS": "vm239",
              "Hematocrit": "vm204",
              "WBC": "vm184",
              "Temp": "vm2",
              "SuppOx": "vm23",
              "SuppFiO2_1": "vm309",
              "SuppFiO2_2": "vm310",
              "PressSupport": "vm211", "MinuteVolume": "vm215", "GCS_Antwort":
              "vm25", "GCS_Motorik": "vm26", "GCS_Augen": "vm27", "SpO2":
              "vm20", "RRate": "vm22", "SaO2": "vm141", "pH": "vm138", "etCO2":
              "vm21", "Total_Bilirubin": "vm162", "Platelets": "vm185",
              "Creatinine": "vm156", "Urine_per_hour": "vm276", "Urine_cum":
              "vm24", "HR": "vm1", "TV": "vm61", "servoi_mode": "vm60",
              "Airway": "vm66", "vent_mode": "vm60", "int_state": "vm312",
              "trach": "vm313",
              "MAP": ["vm5"],
              "Lactate": ["vm136","vm146"],
              "Dobutamine": ["pm41"],
              "Milrinone": ["pm42"],
              "Levosimendan": ["pm43"],
              "Theophyllin": ["pm44"],
              "Norephenephrine": ["pm39"],
              "Epinephrine": ["pm40"],
              "Vasopressin": ["pm45"],
              "Weight": ["vm131"]}

