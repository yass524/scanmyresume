# ats_skills.py
from __future__ import annotations
import re
from typing import Dict, Set, Tuple


# ---------- Master skills dictionary (by category) ----------
SKILLS: Dict[str, Dict[str, Set[str]]] = {
    # ---------- AI / ML ----------
    "ml_ai": {
        "computer vision": {"computer vision","cv","image processing","vision","machine vision"},
        "pytorch": {"pytorch","torch"},
        "tensorflow": {"tensorflow","tf"},
        "opencv": {"opencv","open cv"},
        "onnx": {"onnx"},
        "onnxruntime": {"onnxruntime","onnx runtime"},
        "yolo": {"yolo","yolov5","yolov7","yolov8","yolov9"},
        "transformers": {"transformers","huggingface","hf transformers"},
        "sklearn": {"scikit-learn","sklearn"},
        "xgboost": {"xgboost"},
        "lightgbm": {"lightgbm","lgbm"},
        "catboost": {"catboost"},
        "numpy": {"numpy"},
        "pandas": {"pandas"},
        "mlflow": {"mlflow"},
        "wandb": {"weights & biases","weights and biases","wandb"},
        "inference": {"inference","serving","model serving"},
        "tensorrt": {"tensorrt","trt"},
        "tflite": {"tflite","tensorflow lite"},
        "openvino": {"openvino"},
        "triton": {"triton inference server","triton","nv triton"},
        "langchain": {"langchain"},
        "spacy": {"spacy"},
        "nltk": {"nltk"},
    },

    # ---------- Programming ----------
    "programming": {
        "python": {"python"},
        "c++": {"c++","cpp"},
        "c": {"c"},
        "c#": {"c#",".net","dotnet"},
        "java": {"java"},
        "go": {"go","golang"},
        "rust": {"rust"},
        "scala": {"scala"},
        "matlab": {"matlab"},
        "simulink": {"simulink"},
        "sql": {"sql"},
        "bash": {"bash","shell","sh"},
        "powershell": {"powershell","ps"},
        "javascript": {"javascript","js"},
        "typescript": {"typescript","ts"},
        "r": {"r","r language"},
        "php": {"php"},
        "perl": {"perl"},
    },

    # ---------- Cloud / DevOps ----------
    "cloud_devops": {
        "docker": {"docker","containers","containerization"},
        "kubernetes": {"kubernetes","k8s"},
        "linux": {"linux","ubuntu","debian","centos","rhel"},
        "aws": {"aws","amazon web services","sagemaker","ec2","s3","ecr","lambda"},
        "azure": {"azure","aks","azure devops"},
        "gcp": {"gcp","google cloud","gke","bigquery"},
        "terraform": {"terraform","iac","infrastructure as code"},
        "ansible": {"ansible"},
        "helm": {"helm","helm charts"},
        "prometheus": {"prometheus"},
        "grafana": {"grafana"},
        "ci/cd": {"ci/cd","cicd","github actions","gitlab ci","jenkins"},
    },

    # ---------- Controls / Automation ----------
    "controls_automation": {
        "plc": {"plc","programmable logic controller"},
        "siemens s7": {"siemens s7","s7-1200","s7-1500","tia portal","wincc"},
        "allen-bradley": {"allen-bradley","rockwell","controllogix","studio 5000","rslogix"},
        "schneider electric": {"schneider electric","m340","m580","ecostruxure"},
        "mitsubishi": {"mitsubishi","fx5u","gx works"},
        "omron": {"omron","sysmac","cx-programmer"},
        "beckhoff": {"beckhoff","twincat","ethercat"},
        "codesys": {"codesys"},
        "scada": {"scada","hmi","ignition","factorytalk","wonderware"},
        "vfd": {"vfd","variable frequency drive","drive"},
        "servo": {"servo","servo drive","motion control"},
        "instrumentation": {"instrumentation","sensors","transmitters","analog io","digital io"},
        "modbus": {"modbus","modbus tcp","modbus rtu"},
        "profinet": {"profinet","profibus"},
        "ethernet/ip": {"ethernet/ip","ethernet ip"},
        "opc ua": {"opc ua","opcua"},
        "mqtt": {"mqtt"},
        "cmms": {"cmms","sap pm","maximo"},
        "rca": {"root cause analysis","5-why","fmea"},
    },

    # ---------- Embedded / Edge ----------
    "embedded": {
        "arduino": {"arduino"},
        "esp32": {"esp32","esp-idf"},
        "stm32": {"stm32","stm32cube"},
        "pic": {"pic","mplab","xc8"},
        "raspberry pi": {"raspberry pi","rpi"},
        "freertos": {"freertos"},
        "zephyr": {"zephyr rtos","zephyr"},
        "uart": {"uart","serial"},
        "i2c": {"i2c"},
        "spi": {"spi"},
        "CAN": {"can bus","can-bus","canbus","canopen","controller area network"},
        "pwm": {"pwm"},
    },

    # ---------- Data Engineering ----------
    "data_eng": {
        "airflow": {"airflow"},
        "spark": {"spark","pyspark"},
        "kafka": {"kafka"},
        "flink": {"flink"},
        "dbt": {"dbt"},
        "redshift": {"redshift"},
        "glue": {"glue","aws glue"},
        "snowflake": {"snowflake"},
        "bigquery": {"bigquery"},
        "databricks": {"databricks"},
        "hive": {"hive"},
    },

    # ---------- Web / Frontend / Backend ----------
    "web": {
        "react": {"react"},
        "next.js": {"next.js","nextjs"},
        "vue": {"vue","vue.js","vuejs"},
        "angular": {"angular"},
        "node": {"node","node.js","nodejs"},
        "express": {"express"},
        "django": {"django"},
        "graphql": {"graphql","gql"},
        "rest api": {"rest","rest api"},
    },

    # ---------- Mobile ----------
    "mobile": {
        "android": {"android","kotlin","java (android)"},
        "ios": {"ios","swift"},
        "react native": {"react native"},
        "flutter": {"flutter","dart"},
    },

    # ---------- Databases ----------
    "databases": {
        "postgresql": {"postgresql","postgres","psql"},
        "mysql": {"mysql"},
        "sqlite": {"sqlite"},
        "mongodb": {"mongodb","mongo"},
        "redis": {"redis"},
        "elasticsearch": {"elasticsearch","elastic","es"},
        "cassandra": {"cassandra"},
        "neo4j": {"neo4j","graph db","graph database"},
    },

    # ---------- Analytics / BI ----------
    "analytics_bi": {
        "excel": {"excel","microsoft excel"},
        "power bi": {"power bi","powerbi"},
        "tableau": {"tableau"},
        "looker": {"looker","google data studio"},
        "superset": {"superset"},
        "matplotlib": {"matplotlib"},
        "plotly": {"plotly"},
    },

    # ---------- Product / PM ----------
    "product_pm": {
        "jira": {"jira"},
        "confluence": {"confluence"},
        "trello": {"trello"},
        "asana": {"asana"},
        "notion": {"notion"},
        "miro": {"miro"},
        "agile": {"agile"},
        "scrum": {"scrum"},
        "kanban": {"kanban"},
        "okrs": {"okrs"},
    },

    # ---------- Business / Finance ----------
    "business_finance": {
        "sap": {"sap","sap erp","sap hana"},
        "oracle erp": {"oracle erp"},
        "netsuite": {"netsuite"},
        "quickbooks": {"quickbooks"},
        "salesforce": {"salesforce"},
        "hubspot": {"hubspot"},
        "zoho": {"zoho"},
        "financial modeling": {"financial modeling","valuation"},
        "accounting": {"accounting","bookkeeping"},
        "procurement": {"procurement"},
    },

    # ---------- Architecture / Civil / Mechanical ----------
    "engineering_other": {
        "autocad": {"autocad","auto cad"},
        "solidworks": {"solidworks"},
        "revit": {"revit"},
        "staad": {"staad","staad pro"},
        "ansys": {"ansys"},
        "catia": {"catia"},
        "etabs": {"etabs"},
        "primavera": {"primavera","p6"},
        "ms project": {"ms project","microsoft project"},
        "hvac": {"hvac"},
        "bim": {"bim","building information modeling"},
        "sap2000": {"sap2000"},
    },

    # ---------- Design / UX ----------
    "design_ux": {
        "figma": {"figma"},
        "adobe xd": {"adobe xd"},
        "illustrator": {"illustrator","adobe illustrator"},
        "photoshop": {"photoshop","adobe photoshop"},
        "ux research": {"ux research","user research"},
        "wireframing": {"wireframing"},
        "prototyping": {"prototyping"},
        "indesign": {"indesign","adobe indesign"},
        "after effects": {"after effects","ae"},
        "premiere": {"premiere","premiere pro"},
    },
}

# ---------- Precompiled alias maps ----------
ALIAS_RE: Dict[str, Tuple[str, re.Pattern]] = {}   # alias -> (canonical, compiled pattern)
CANON_CATEGORY: Dict[str, str] = {}                 # canonical -> category
CANON_ALIASES: Dict[str, Set[str]] = {}             # canonical -> aliases (including canonical)

for cat, items in SKILLS.items():
    for canonical, aliases in items.items():
        CANON_CATEGORY[canonical] = cat
        CANON_ALIASES[canonical] = set(aliases) | {canonical}
        for alias in CANON_ALIASES[canonical]:
            # strict word boundary match, case-insensitive
            ALIAS_RE[alias] = (canonical, re.compile(rf"\b{re.escape(alias)}\b", re.I))
