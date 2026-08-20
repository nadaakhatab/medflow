"""
MedFlow Medical Corpus - Section-Aware Guidelines Dataset
Contains structured clinical guideline chunks from ATA, NIDDK, and Mayo Clinic datasets.
Includes specialized guidelines for Post-Thyroidectomy monitoring and Congenital Hypothyroidism.
"""

from typing import List, Dict, Any

MEDICAL_CORPUS: List[Dict[str, Any]] = [
    {
        "chunk_id": "ATA_HYPO_001",
        "document_name": "ATA_Thyroid_Guidelines_2023.pdf",
        "section": "Section 4.2: Primary Hypothyroidism Diagnostic Protocol",
        "page_number": 14,
        "text_content": (
            "Serum TSH measurement is the primary screening parameter for thyroid dysfunction. "
            "A serum TSH concentration exceeding 4.5 mIU/L with sub-normal serum free T4 (FT4 < 0.82 ng/dL) "
            "confirms primary hypothyroidism. Anti-thyroid peroxidase (anti-TPO) antibody testing is recommended "
            "to establish an underlying autoimmune etiology such as Hashimoto thyroiditis."
        ),
        "disease_category": "Hypothyroidism",
        "keywords": ["TSH", "FT4", "hypothyroidism", "anti-TPO", "Hashimoto", "screening"]
    },
    {
        "chunk_id": "ATA_HYPO_002",
        "document_name": "ATA_Thyroid_Guidelines_2023.pdf",
        "section": "Section 5.1: Subclinical Hypothyroidism & Levothyroxine Titration",
        "page_number": 19,
        "text_content": (
            "Subclinical hypothyroidism is defined as an elevated TSH (4.5–10 mIU/L) with normal free T4 levels. "
            "Routine levothyroxine (L-T4) replacement therapy is strongly indicated when TSH exceeds 10 mIU/L, "
            "or in symptomatic patients with positive anti-TPO antibodies or elevated cardiovascular risk. "
            "Initial daily dosing of Levothyroxine is estimated at 1.6 mcg/kg ideal body weight, re-evaluating serum TSH in 6–8 weeks."
        ),
        "disease_category": "Hypothyroidism",
        "keywords": ["subclinical", "Levothyroxine", "L-T4", "TSH", "dosing", "titration"]
    },
    {
        "chunk_id": "ATA_PREG_001",
        "document_name": "ATA_Thyroid_Guidelines_2023.pdf",
        "section": "Section 8.4: Gestational Thyroid Management & Pregnancy Targets",
        "page_number": 42,
        "text_content": (
            "During the first trimester of pregnancy, upper reference limit for TSH is 2.5 mIU/L (or trimester-specific reference range). "
            "Maternal subclinical or overt hypothyroidism increases risk of pregnancy loss, premature delivery, and impaired neurodevelopment. "
            "Subclinical hypothyroidism in pregnancy with positive anti-TPO antibodies warrants immediate levothyroxine intervention."
        ),
        "disease_category": "Pregnancy & Thyroid",
        "keywords": ["pregnancy", "trimester", "gestational", "TSH", "anti-TPO", "neurodevelopment"]
    },
    {
        "chunk_id": "NIDDK_GRAVES_001",
        "document_name": "NIDDK_Thyroid_Overview.pdf",
        "section": "Section 3.1: Pathophysiology of Graves' Disease & Hyperthyroidism",
        "page_number": 28,
        "text_content": (
            "Graves' disease is an autoimmune condition caused by thyroid-stimulating immunoglobulin (TSI) autoantibodies "
            "activating the TSH receptor, resulting in autonomous thyroid hormone hypersecretion. "
            "Clinical presentation includes suppressed TSH (<0.1 mIU/L), elevated free T4 (>1.77 ng/dL), elevated free T3, "
            "diffuse thyroid enlargement (goiter), orbitopathy (exophthalmos), and pretibial myxedema."
        ),
        "disease_category": "Hyperthyroidism",
        "keywords": ["Graves", "TSI", "hyperthyroidism", "goiter", "exophthalmos", "anti-TSHR"]
    },
    {
        "chunk_id": "NIDDK_GRAVES_002",
        "document_name": "NIDDK_Thyroid_Overview.pdf",
        "section": "Section 3.4: Antithyroid Pharmacotherapy (MMI vs PTU)",
        "page_number": 33,
        "text_content": (
            "Methimazole (MMI) is the first-line antithyroid drug choice for Graves' hyperthyroidism due to lower hepatotoxicity risk. "
            "Propylthiouracil (PTU) is preferred specifically during the first trimester of pregnancy due to MMI embryopathy concerns, "
            "and in acute treatment of thyroid storm. Complete blood counts must be monitored for agranulocytosis risk."
        ),
        "disease_category": "Hyperthyroidism",
        "keywords": ["Methimazole", "MMI", "PTU", "Propylthiouracil", "pregnancy", "thyroid storm", "agranulocytosis"]
    },
    {
        "chunk_id": "MAYO_NODULE_001",
        "document_name": "Mayo_Clinic_Endocrine_Manual.pdf",
        "section": "Section 6.2: Thyroid Nodule Evaluation & Fine Needle Aspiration (FNA)",
        "page_number": 112,
        "text_content": (
            "Thyroid nodules measuring >1.0 cm with high-risk US features (hypoechoic, microcalcifications, irregular margins, tall-than-wide shape) "
            "should undergo ultrasound-guided Fine-Needle Aspiration (FNA) biopsy. "
            "Cytopathology reporting follows the Bethesda System for Reporting Thyroid Cytopathology (Categories I through VI)."
        ),
        "disease_category": "Nodules & Cancer",
        "keywords": ["nodule", "ultrasound", "FNA", "biopsy", "Bethesda", "microcalcifications"]
    },
    {
        "chunk_id": "MAYO_HASHI_001",
        "document_name": "Mayo_Clinic_Endocrine_Manual.pdf",
        "section": "Section 2.3: Chronic Lymphocytic Thyroiditis (Hashimoto)",
        "page_number": 88,
        "text_content": (
            "Hashimoto's thyroiditis is the most common cause of hypothyroidism in iodine-sufficient areas. "
            "Histopathology demonstrates marked lymphocytic infiltration with germinal centers and Hurthle cell metaplasia. "
            "Diagnostic confirmation relies on serum anti-TPO (>34 IU/mL) and anti-thyroglobulin (anti-Tg) antibody titers."
        ),
        "disease_category": "Hypothyroidism",
        "keywords": ["Hashimoto", "lymphocytic", "anti-TPO", "anti-Tg", "Hurthle", "autoimmune"]
    },
    {
        "chunk_id": "ATA_THYROIDITIS_001",
        "document_name": "ATA_Thyroid_Guidelines_2023.pdf",
        "section": "Section 7.1: Subacute (De Quervain's) Thyroiditis Clinical Course",
        "page_number": 56,
        "text_content": (
            "Subacute granulomatous thyroiditis typically presents post-viral with anterior neck pain, fever, and transient hyperthyroidism "
            "due to follicular disruption. Radioactive iodine uptake (RAIU) is markedly low (<1%), distinguishing it from Graves' disease. "
            "Treatment focuses on NSAIDs for mild cases and systemic corticosteroids for severe pain."
        ),
        "disease_category": "Thyroiditis",
        "keywords": ["subacute", "De Quervain", "neck pain", "RAIU", "corticosteroids", "transient"]
    },

    # --- SPECIALIZED POST-THYROIDECTOMY GUIDELINES ---
    {
        "chunk_id": "ATA_SURG_001",
        "document_name": "ATA_Surgical_Guidelines_2024.pdf",
        "section": "Section 9.1: Post-Thyroidectomy Hormone Replacement & TSH Targets",
        "page_number": 64,
        "text_content": (
            "In athyreotic patients post-total thyroidectomy or radioiodine ablation for benign disease, serum TSH target is "
            "maintained within the normal physiological reference range (0.5–2.5 mIU/L). An elevated TSH in an athyreotic patient "
            "reflects insufficient exogenous levothyroxine substitution or reduced bioavailability. Malabsorption factors include "
            "co-administration of calcium carbonate, ferrous sulfate, aluminum hydroxide, or proton pump inhibitors, which must be "
            "spaced at least 4 hours apart from levothyroxine ingestion."
        ),
        "disease_category": "Post-Thyroidectomy",
        "keywords": ["thyroidectomy", "athyreotic", "ablation", "levothyroxine", "malabsorption", "calcium", "iron", "spacing"]
    },
    {
        "chunk_id": "ATA_CANCER_001",
        "document_name": "ATA_Thyroid_Cancer_Guidelines_2024.pdf",
        "section": "Section 11.3: Post-Operative TSH Suppression Targets & Tumor Markers",
        "page_number": 95,
        "text_content": (
            "Following total thyroidectomy and radioiodine ablation for differentiated thyroid cancer (papillary or follicular), "
            "TSH suppression targets are tailored by disease risk stratification: high-risk recurrence requires TSH suppression < 0.1 mIU/L, "
            "intermediate-risk targets TSH 0.1–0.5 mIU/L, and low-risk disease targets TSH 0.5–2.0 mIU/L. Serum thyroglobulin (Tg) and "
            "anti-thyroglobulin antibodies (anti-Tg) serve as key post-operative tumor markers. Serum Tg must always be interpreted alongside "
            "anti-Tg titers, as positive anti-Tg autoantibodies can falsely suppress Tg measurements."
        ),
        "disease_category": "Thyroid Cancer",
        "keywords": ["cancer", "thyroidectomy", "suppression", "TSH", "thyroglobulin", "Tg", "anti-Tg", "recurrence"]
    },
    {
        "chunk_id": "MAYO_SURG_001",
        "document_name": "Mayo_Clinic_Surgical_Manual.pdf",
        "section": "Section 10.2: Post-Thyroidectomy Hypocalcemia & Parathyroid Function",
        "page_number": 142,
        "text_content": (
            "Transient hypocalcemia is a recognized post-thyroidectomy surgical risk due to parathyroid stunned state or inadvertent "
            "devascularization. Serum total calcium (<8.0 mg/dL) and intact parathyroid hormone (PTH <15 pg/mL) should be monitored. "
            "Clinical neuromuscular irritability features include circumoral numbness, paresthesias, and positive Chvostek or Trousseau signs, "
            "warranting prompt oral calcium carbonate and calcitriol supplementation under clinical supervision."
        ),
        "disease_category": "Post-Thyroidectomy",
        "keywords": ["hypocalcemia", "parathyroid", "PTH", "calcium", "Chvostek", "Trousseau", "surgery"]
    },

    # --- SPECIALIZED CONGENITAL HYPOTHYROIDISM GUIDELINES ---
    {
        "chunk_id": "ATA_PED_001",
        "document_name": "ATA_LWPES_Pediatric_Guidelines.pdf",
        "section": "Section 2.1: Congenital Hypothyroidism Diagnostic Protocol & Reference Range Safety",
        "page_number": 12,
        "text_content": (
            "Congenital hypothyroidism (thyroid agenesis, athyreosis, dysgenesis, ectopic thyroid, or dyshormonogenesis) requires prompt "
            "initiation of oral levothyroxine (10–15 mcg/kg/day) within the first 14 days of life to prevent permanent neurodevelopmental impairment. "
            "CRITICAL SAFETY RULE: Standard adult reference ranges (e.g. 0.45–4.5 mIU/L) MUST NEVER be applied to pediatric patients or infants. "
            "Serum TSH and Free T4 must be evaluated exclusively against age-specific pediatric reference standards (e.g. newborn TSH surge "
            "up to 20 mIU/L in first 24–48 hours, normalizing rapidly thereafter; infant target FT4 upper half of reference range 1.4–2.3 ng/dL)."
        ),
        "disease_category": "Congenital Hypothyroidism",
        "keywords": ["congenital", "agenesis", "athyreosis", "pediatric", "infant", "newborn", "reference range", "neurodevelopment"]
    },
    {
        "chunk_id": "ATA_PED_002",
        "document_name": "ATA_LWPES_Pediatric_Guidelines.pdf",
        "section": "Section 3.4: Congenital Hypothyroidism Follow-Up & Growth Monitoring",
        "page_number": 24,
        "text_content": (
            "Children with congenital hypothyroidism require frequent laboratory monitoring: every 1–2 months during the first 6 months of life, "
            "every 2–3 months from 6 months to 3 years of age, and every 3–6 months thereafter until linear growth is completed. "
            "Monitoring parameters include TSH, Free T4, linear growth velocity, weight gain, and neurodevelopmental milestones. "
            "Levothyroxine replacement dosing must be dynamically adjusted as child body weight increases."
        ),
        "disease_category": "Congenital Hypothyroidism",
        "keywords": ["congenital", "follow-up", "monitoring", "pediatric", "growth", "milestones", "weight"]
    }
]
