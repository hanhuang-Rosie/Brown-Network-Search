import pandas as pd
import streamlit as st

# 1. Page setup

st.set_page_config(
    page_title="Brown Network Search Prototype",
    layout="wide"
)


# 2. Helper functions

def clean_text(value):
    """
    Clean text for matching.
    Example:
    ' Access Holdings ' -> 'access holdings'
    """
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def calculate_score(features):
    """
    Calculate score contribution for one relationship evidence.
    """
    weights = {
        "direct_io_contact": 40,
        "current_employer_alumni_overlap": 25,
        "current_employer_io_overlap": 22,
        "previous_employer_alumni_overlap": 18,
        "previous_employer_io_overlap": 15,
        "managed_alumni_contact": 10
    }

    score = 0

    for feature, value in features.items():
        score += weights.get(feature, 0) * value

    return score


@st.cache_data
def load_data():

    alumni_df = pd.read_csv("/Users/huanghan/Desktop/Contacts and Network/mock_alumni.csv")
    io_contacts_df = pd.read_csv("/Users/huanghan/Desktop/Contacts and Network/mock_io_contacts.csv")

    return alumni_df, io_contacts_df


# 3. Existing person suggestions

def get_existing_person_suggestions(query, alumni_df, io_contacts_df):
    """
    When the user types a full name, this checks whether similar names
    already exist in Brown alumni data or IO contact data.
    """
    if not query:
        return pd.DataFrame(columns=["name", "source", "company", "title"])

    q = clean_text(query)
    records = []

    # Search Brown alumni
    for _, row in alumni_df.iterrows():
        name = str(row.get("Full Name", ""))

        if q in clean_text(name):
            records.append({
                "name": name,
                "source": "Brown Alumni Data",
                "company": row.get("Primary Employer", ""),
                "title": row.get("Job Title", "")
            })

    # Search IO contacts
    for _, row in io_contacts_df.iterrows():
        name = str(row.get("Contact Full Name", ""))

        if q in clean_text(name):
            records.append({
                "name": name,
                "source": "IO Contact Data",
                "company": row.get("Company", ""),
                "title": row.get("Position", "")
            })

    suggestion_df = pd.DataFrame(records)

    if not suggestion_df.empty:
        suggestion_df = suggestion_df.drop_duplicates()

    return suggestion_df


# 4. Main matching function

def search_new_person(
    full_name,
    current_company,
    previous_company,
    alumni_df,
    io_contacts_df
):
    """
    Search one external person against:
    1. Brown alumni data
    2. IO contacts data

    Each matched relationship becomes one evidence row.
    Each evidence row has score_contribution.
    Later, all score_contribution values are summed into total_score.
    """

    results = []
    relationship_id = 1

    full_name_clean = clean_text(full_name)
    current_company_clean = clean_text(current_company)
    previous_company_clean = clean_text(previous_company)

    # Match 1: Direct IO LinkedIn Contact
    # Full Name = IO Contact Full Name

    direct_io_matches = io_contacts_df[
        io_contacts_df["Contact Full Name"].apply(clean_text) == full_name_clean
    ]

    for _, contact in direct_io_matches.iterrows():
        features = {
            "direct_io_contact": 1,
            "current_employer_alumni_overlap": 0,
            "current_employer_io_overlap": 0,
            "previous_employer_alumni_overlap": 0,
            "previous_employer_io_overlap": 0,
            "managed_alumni_contact": 0
        }

        score = calculate_score(features)

        results.append({
            "relationship_id": f"R{relationship_id:03d}",
            "target_person": full_name,
            "match_type": "Direct IO LinkedIn Contact",
            "brown_related_contact": contact["Contact Full Name"],
            "related_io_member": contact["IO Member"],
            "relationship_path": (
                f"{full_name} appears in the IO contact list and is connected to "
                f"{contact['IO Member']}."
            ),
            "evidence": "Name matched IO LinkedIn contact export",
            "confidence": "High",
            "score_contribution": score,
            **features
        })

        relationship_id += 1

    # Match 2: Current Company x Brown Alumni
    # Current Company = Alumni Primary Employer

    if current_company_clean:
        alumni_current_matches = alumni_df[
            alumni_df["Primary Employer"].apply(clean_text) == current_company_clean
        ]

        for _, alum in alumni_current_matches.iterrows():
            managed_flag = 1 if clean_text(alum.get("Managed", "")) == "yes" else 0

            features = {
                "direct_io_contact": 0,
                "current_employer_alumni_overlap": 1,
                "current_employer_io_overlap": 0,
                "previous_employer_alumni_overlap": 0,
                "previous_employer_io_overlap": 0,
                "managed_alumni_contact": managed_flag
            }

            score = calculate_score(features)

            results.append({
                "relationship_id": f"R{relationship_id:03d}",
                "target_person": full_name,
                "match_type": "Current Employer Alumni Overlap",
                "brown_related_contact": alum["Full Name"],
                "related_io_member": "",
                "relationship_path": (
                    f"{full_name} works at {current_company}; "
                    f"Brown alum {alum['Full Name']} also works at "
                    f"{alum['Primary Employer']}."
                ),
                "evidence": "Current company matched Alumni Primary Employer",
                "confidence": "Medium",
                "score_contribution": score,
                **features
            })

            relationship_id += 1

    # Match 3: Current Company x IO Contacts
    # Current Company = IO Contact Company

    if current_company_clean:
        io_current_matches = io_contacts_df[
            io_contacts_df["Company"].apply(clean_text) == current_company_clean
        ]
        io_current_matches = io_current_matches[
            io_current_matches["Contact Full Name"].apply(clean_text) != full_name_clean
        ]
        for _, contact in io_current_matches.iterrows():
            features = {
                "direct_io_contact": 0,
                "current_employer_alumni_overlap": 0,
                "current_employer_io_overlap": 1,
                "previous_employer_alumni_overlap": 0,
                "previous_employer_io_overlap": 0,
                "managed_alumni_contact": 0
            }

            score = calculate_score(features)

            results.append({
                "relationship_id": f"R{relationship_id:03d}",
                "target_person": full_name,
                "match_type": "Current Employer IO Contact Overlap",
                "brown_related_contact": contact["Contact Full Name"],
                "related_io_member": contact["IO Member"],
                "relationship_path": (
                    f"{full_name} works at {current_company}; "
                    f"{contact['IO Member']} has a LinkedIn contact at the same company: "
                    f"{contact['Contact Full Name']}."
                ),
                "evidence": "Current company matched IO contact company",
                "confidence": "Medium",
                "score_contribution": score,
                **features
            })

            relationship_id += 1

    # Match 4: Previous Company x Brown Alumni
    # Previous Company = Alumni Primary Employer

    if previous_company_clean:
        alumni_previous_matches = alumni_df[
            alumni_df["Primary Employer"].apply(clean_text) == previous_company_clean
        ]

        for _, alum in alumni_previous_matches.iterrows():
            managed_flag = 1 if clean_text(alum.get("Managed", "")) == "yes" else 0

            features = {
                "direct_io_contact": 0,
                "current_employer_alumni_overlap": 0,
                "current_employer_io_overlap": 0,
                "previous_employer_alumni_overlap": 1,
                "previous_employer_io_overlap": 0,
                "managed_alumni_contact": managed_flag
            }

            score = calculate_score(features)

            results.append({
                "relationship_id": f"R{relationship_id:03d}",
                "target_person": full_name,
                "match_type": "Previous Employer Alumni Overlap",
                "brown_related_contact": alum["Full Name"],
                "related_io_member": "",
                "relationship_path": (
                    f"{full_name} previously worked at {previous_company}; "
                    f"Brown alum {alum['Full Name']} currently works at "
                    f"{alum['Primary Employer']}."
                ),
                "evidence": "Previous company matched Alumni Primary Employer",
                "confidence": "Medium",
                "score_contribution": score,
                **features
            })

            relationship_id += 1

    # Match 5: Previous Company x IO Contacts
    # Previous Company = IO Contact Company

    if previous_company_clean:
        io_previous_matches = io_contacts_df[
            io_contacts_df["Company"].apply(clean_text) == previous_company_clean
        ]

        for _, contact in io_previous_matches.iterrows():
            features = {
                "direct_io_contact": 0,
                "current_employer_alumni_overlap": 0,
                "current_employer_io_overlap": 0,
                "previous_employer_alumni_overlap": 0,
                "previous_employer_io_overlap": 1,
                "managed_alumni_contact": 0
            }

            score = calculate_score(features)

            results.append({
                "relationship_id": f"R{relationship_id:03d}",
                "target_person": full_name,
                "match_type": "Previous Employer IO Contact Overlap",
                "brown_related_contact": contact["Contact Full Name"],
                "related_io_member": contact["IO Member"],
                "relationship_path": (
                    f"{full_name} previously worked at {previous_company}; "
                    f"{contact['IO Member']} has a LinkedIn contact at that company: "
                    f"{contact['Contact Full Name']}."
                ),
                "evidence": "Previous company matched IO contact company",
                "confidence": "Medium",
                "score_contribution": score,
                **features
            })

            relationship_id += 1

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        result_df = result_df.sort_values(
            by="score_contribution",
            ascending=False
        )

    return result_df

# # 5. Load data
# alumni_df, io_contacts_df = load_data()

# # 6. Dashboard header

# st.title("Brown Network Search Prototype")

# st.write(
#     "This MVP searches whether an external investment professional has possible "
#     "relationship paths to the Brown network through Brown alumni data and "
#     "Investment Office contact data."
# )


# # 7. Summary metrics

# col1, col2, col3 = st.columns(3)

# col1.metric("Brown Alumni", len(alumni_df))
# col2.metric("IO Contacts", len(io_contacts_df))
# col3.metric("Data Pools", 2)

# st.divider()


# # 8. New person search form

# st.subheader("Search a New External Person")

# with st.form("new_person_search"):
#     full_name = st.text_input("Full Name")
#     current_company = st.text_input("Current Company")
#     previous_company = st.text_input("Previous Company")

#     submitted = st.form_submit_button("Find Brown Network Connections")


# # 9. Possible existing matches

# suggestion_df = get_existing_person_suggestions(
#     full_name,
#     alumni_df,
#     io_contacts_df
# )

# if full_name and not suggestion_df.empty:
#     st.markdown("#### Possible Matches in Current Data Pool")
#     st.dataframe(
#         suggestion_df,
#         use_container_width=True,
#         hide_index=True
#     )

# elif full_name:
#     st.info(
#         "No exact person match found in the current data pool. "
#         "The tool will continue searching by company overlap."
#     )


# # 10. Relationship results

# if submitted:
#     if not full_name:
#         st.warning("Please enter at least a full name.")

#     else:
#         result_df = search_new_person(
#             full_name=full_name,
#             current_company=current_company,
#             previous_company=previous_company,
#             alumni_df=alumni_df,
#             io_contacts_df=io_contacts_df
#         )

#         st.divider()
#         st.subheader("Relationship Search Results")

#         if result_df.empty:
#             st.info(
#                 "No Brown network relationship path found based on the current inputs."
#             )

#         else:
#             # Overall Match Summary

#             total_score = min(result_df["score_contribution"].sum(), 100)

#             if total_score >= 80:
#                 overall_confidence = "High"
#             elif total_score >= 50:
#                 overall_confidence = "Medium"
#             else:
#                 overall_confidence = "Low"

#             st.markdown("### Overall Match Summary")

#             summary_col1, summary_col2, summary_col3 = st.columns(3)

#             summary_col1.metric("Target Person", full_name)
#             summary_col2.metric("Total Score", int(total_score))
#             summary_col3.metric("Overall Confidence", overall_confidence)

#             st.progress(int(total_score))

#             st.write(
#                 "The total score is the sum of all relationship evidence scores, "
#                 "capped at 100."
#             )

#             # Relationship Evidence Cards

#             st.markdown("### Relationship Evidence Details")

#             for _, row in result_df.iterrows():
#                 with st.container(border=True):
#                     st.markdown(f"#### {row['match_type']}")

#                     st.write("**Target Person:**", row["target_person"])
#                     st.write("**Brown-related Contact:**", row["brown_related_contact"])

#                     if row["related_io_member"]:
#                         st.write("**Related IO Member:**", row["related_io_member"])
#                     else:
#                         st.write("**Related IO Member:** N/A")

#                     st.info(row["relationship_path"])

#                     st.write("**Confidence:**", row["confidence"])
#                     st.write("**Score Contribution:**", int(row["score_contribution"]))
#                     st.progress(min(int(row["score_contribution"]), 100))
#                     st.write("**Evidence:**", row["evidence"])

#             # Ranked Evidence Table
#             st.markdown("### Ranked Evidence Table")

#             st.dataframe(
#                 result_df,
#                 use_container_width=True,
#                 hide_index=True
#             )


# # 11. Score explanation

# st.divider()

# with st.expander("How is the ranking score calculated?"):
#     st.write(
#         "The prototype uses a feature-weighted rule-based scoring system. "
#         "The score does not evaluate the quality of a person or organization. "
#         "It measures the relative strength of possible relationship paths."
#     )

#     score_table = pd.DataFrame([
#         {
#             "Feature": "Direct IO LinkedIn Contact",
#             "Weight": 40,
#             "Meaning": "The person appears directly in an IO member's contact list."
#         },
#         {
#             "Feature": "Current Employer Alumni Overlap",
#             "Weight": 25,
#             "Meaning": "The person's current company matches a Brown alum's current employer."
#         },
#         {
#             "Feature": "Current Employer IO Contact Overlap",
#             "Weight": 22,
#             "Meaning": "The person's current company matches an IO contact's company."
#         },
#         {
#             "Feature": "Previous Employer Alumni Overlap",
#             "Weight": 18,
#             "Meaning": "The person's previous company matches a Brown alum's current employer."
#         },
#         {
#             "Feature": "Previous Employer IO Contact Overlap",
#             "Weight": 15,
#             "Meaning": "The person's previous company matches an IO contact's company."
#         },
#         {
#             "Feature": "Managed Alumni Contact",
#             "Weight": 10,
#             "Meaning": "The matched Brown alum is marked as a managed relationship."
#         }
#     ])

#     st.dataframe(
#         score_table,
#         use_container_width=True,
#         hide_index=True
#     )

# # 12. Data preview

# st.divider()

# st.subheader("Data Preview")

# with st.expander("Brown Alumni Data"):
#     st.dataframe(
#         alumni_df,
#         use_container_width=True,
#         hide_index=True
#     )

# with st.expander("Investment Office Contact Data"):
#     st.dataframe(
#         io_contacts_df,
#         use_container_width=True,
#         hide_index=True
#     )

# ============================================================
# 5. Load data
# ============================================================

alumni_df, io_contacts_df = load_data()


# ============================================================
# 6. Custom CSS
# ============================================================

st.markdown("""
<style>
/* App background */
.stApp {
    background-color: #f7f8fb;
}

/* Main content */
.block-container {
    padding-top: 2.2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1400px;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #edf0f5;
}

/* Sidebar inner padding */
section[data-testid="stSidebar"] > div {
    padding-top: 1.0rem;
    padding-left: 0.75rem;
    padding-right: 0.75rem;
}

/* Sidebar brand */
.sidebar-brand {
    font-size: 1.25rem;
    font-weight: 800;
    color: #242938;
    margin-top: 0.4rem;
    margin-bottom: 0.1rem;
}

.sidebar-subtitle {
    font-size: 0.82rem;
    color: #7b8190;
    margin-bottom: 1.4rem;
}

/* Sidebar section title */
.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 800;
    color: #9aa3b2;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 1.0rem;
    margin-bottom: 0.45rem;
}

/* Sidebar button style: ChatGPT-like */
section[data-testid="stSidebar"] .stButton > button {
    background-color: transparent;
    border: none;
    color: #2f3342;
    font-size: 0.95rem;
    font-weight: 500;
    text-align: left;
    justify-content: flex-start;
    padding: 0.62rem 0.75rem;
    border-radius: 10px;
    transition: background-color 0.12s ease-in-out;
}

/* Hover effect */
section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #f2f3f5;
    color: #2f3342;
}

/* Remove default focus border */
section[data-testid="stSidebar"] .stButton > button:focus {
    box-shadow: none;
    border: none;
}

/* Current page label */
.current-page-pill {
    background-color: #eef0f4;
    color: #2f3342;
    padding: 0.6rem 0.75rem;
    border-radius: 10px;
    font-size: 0.88rem;
    font-weight: 650;
    margin-top: 0.4rem;
    margin-bottom: 0.9rem;
}

/* Sidebar summary cards */
.sidebar-summary-card {
    background-color: #f7f8fb;
    border: 1px solid #edf0f5;
    border-radius: 12px;
    padding: 0.75rem 0.8rem;
    margin-bottom: 0.6rem;
}

.sidebar-summary-label {
    color: #7b8190;
    font-size: 0.75rem;
    margin-bottom: 0.15rem;
}

.sidebar-summary-value {
    color: #242938;
    font-size: 1rem;
    font-weight: 750;
}

/* Sidebar footer */
.sidebar-footer {
    color: #9aa3b2;
    font-size: 0.76rem;
    margin-top: 1.2rem;
}

/* Page title */
.page-title {
    font-size: 2.4rem;
    font-weight: 850;
    color: #242938;
    margin-bottom: 0.35rem;
}

.page-subtitle {
    font-size: 1rem;
    color: #5f6675;
    margin-bottom: 1.5rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #edf0f5;
    border-radius: 18px;
    padding: 1rem 1.2rem;
    box-shadow: 0 8px 24px rgba(31, 41, 55, 0.04);
}

/* General buttons */
.stButton > button {
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
}

/* Inputs */
.stTextInput > div > div > input {
    border-radius: 12px;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 7. Sidebar Navigation - No URL jump version
# ============================================================

if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"


def sidebar_nav_button(label, icon, target):
    clicked = st.sidebar.button(
        f"{icon}  {label}",
        key=f"nav_{target}",
        use_container_width=True
    )

    if clicked:
        st.session_state.current_page = target
        st.rerun()


with st.sidebar:
    # -----------------------------
    # Logo + brand
    # -----------------------------

    logo_path = "/Users/huanghan/Desktop/Contacts and Network/Brown-University-Logo.png"

    st.image(
        logo_path,
        width=42
    )

    st.markdown(
        """
        <div class="sidebar-brand">Brown Network</div>
        <div class="sidebar-subtitle">Search Prototype</div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Navigation
    # -----------------------------

    st.markdown(
        '<div class="sidebar-section-title">Main Navigation</div>',
        unsafe_allow_html=True
    )

    sidebar_nav_button("Dashboard", "▦", "Dashboard")
    sidebar_nav_button("Search & Results", "⌕", "SearchResults")
    sidebar_nav_button("Data Sources", "◫", "DataSources")
    sidebar_nav_button("Scoring Logic", "◇", "ScoringLogic")

    current_page = st.session_state.current_page

    page_display_name = {
        "Dashboard": "Dashboard",
        "SearchResults": "Search & Results",
        "DataSources": "Data Sources",
        "ScoringLogic": "Scoring Logic"
    }.get(current_page, "Dashboard")

    st.markdown(
        f"""
        <div class="current-page-pill">
            Current: {page_display_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------
    # Data summary
    # -----------------------------

    st.markdown(
        '<div class="sidebar-section-title">Data Summary</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="sidebar-summary-card">
            <div class="sidebar-summary-label">Brown Alumni</div>
            <div class="sidebar-summary-value">{len(alumni_df)}</div>
        </div>

        <div class="sidebar-summary-card">
            <div class="sidebar-summary-label">IO Contacts</div>
            <div class="sidebar-summary-value">{len(io_contacts_df)}</div>
        </div>

        <div class="sidebar-summary-card">
            <div class="sidebar-summary-label">Data Pools</div>
            <div class="sidebar-summary-value">2</div>
        </div>

        <div class="sidebar-footer">
            Prototype v1.0<br>
            Mock data mode
        </div>
        """,
        unsafe_allow_html=True
    )


current_page = st.session_state.current_page


# ============================================================
# 8. Dashboard Page
# ============================================================

if current_page == "Dashboard":

    st.markdown(
        """
        <div class="page-title">Brown Network Search Dashboard</div>
        <div class="page-subtitle">
            Overview of available data pools and relationship discovery signals.
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Brown Alumni", len(alumni_df))
    col2.metric("IO Contacts", len(io_contacts_df))
    col3.metric("Data Pools", 2)
    col4.metric("Search Mode", "External Person")

    st.markdown("<br>", unsafe_allow_html=True)

    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown("### Data Pool Overview")

        data_pool_overview = pd.DataFrame([
            {
                "Data Pool": "Brown Alumni Data",
                "Description": "Brown alumni with employer and relationship information",
                "Key Matching Fields": "Full Name, Primary Employer, Managed"
            },
            {
                "Data Pool": "Investment Office Contact Data",
                "Description": "LinkedIn contact exports from Investment Office members",
                "Key Matching Fields": "Contact Full Name, Company, IO Member"
            }
        ])

        st.dataframe(
            data_pool_overview,
            use_container_width=True,
            hide_index=True
        )

    with right_col:
        st.markdown("### Prototype Goal")

        st.info(
            "Search an external investment professional and identify possible "
            "relationship paths through Brown alumni and IO contact data."
        )

        st.markdown("#### Core Search Inputs")
        st.write("- Full Name")
        st.write("- Current Company")
        st.write("- Previous Company")

    st.divider()

    st.markdown("### Relationship Signals")

    signal_table = pd.DataFrame([
        {
            "Relationship Signal": "Direct IO LinkedIn Contact",
            "Meaning": "The person appears directly in an IO member's contact list",
            "Strength": "Strong"
        },
        {
            "Relationship Signal": "Current Employer Alumni Overlap",
            "Meaning": "The person's current company matches a Brown alum's employer",
            "Strength": "Medium"
        },
        {
            "Relationship Signal": "Current Employer IO Contact Overlap",
            "Meaning": "The person's current company matches an IO contact's company",
            "Strength": "Medium"
        },
        {
            "Relationship Signal": "Previous Employer Alumni Overlap",
            "Meaning": "The person's previous company matches a Brown alum's employer",
            "Strength": "Medium"
        },
        {
            "Relationship Signal": "Previous Employer IO Contact Overlap",
            "Meaning": "The person's previous company matches an IO contact's company",
            "Strength": "Medium-Low"
        }
    ])

    st.dataframe(
        signal_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Example Search Cases")

    example_cases = pd.DataFrame([
        {
            "Full Name": "Katie Leshinsky",
            "Current Company": "DFW Capital Partners",
            "Previous Company": "Access Holdings",
            "Expected Result": "Multiple direct and employer-overlap paths"
        },
        {
            "Full Name": "Omar Rahman",
            "Current Company": "Access Holdings",
            "Previous Company": "",
            "Expected Result": "Direct IO contact and employer overlaps"
        },
        {
            "Full Name": "Kevin Cuthbert",
            "Current Company": "Access Holdings",
            "Previous Company": "",
            "Expected Result": "Not in data pool, but company overlaps found"
        },
        {
            "Full Name": "Kim Kile",
            "Current Company": "Victor Capital Partners",
            "Previous Company": "Access Holdings",
            "Expected Result": "Current and previous employer overlaps"
        }
    ])

    st.dataframe(
        example_cases,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 9. Search & Results Page
# ============================================================

elif current_page == "SearchResults":

    st.markdown(
        """
        <div class="page-title">Search & Results</div>
        <div class="page-subtitle">
            Enter an external professional's information to identify possible Brown network connections.
        </div>
        """,
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([1.35, 1])

    with left_col:
        st.markdown("### Search a New External Person")

        with st.form("new_person_search"):
            full_name = st.text_input("Full Name")
            current_company = st.text_input("Current Company")
            previous_company = st.text_input("Previous Company")

            submitted = st.form_submit_button("Find Brown Network Connections")

    with right_col:
        st.markdown("### Search Guidance")

        st.info(
            "Use company fields to identify employer overlaps with Brown alumni "
            "and IO contacts. Name matching checks whether the person already "
            "appears in the current data pool."
        )

        st.markdown("#### Demo examples")
        st.write("- Katie Leshinsky")
        st.write("- Omar Rahman")
        st.write("- Kevin Cuthbert")
        st.write("- Kim Kile")

    if "full_name" in locals():
        suggestion_df = get_existing_person_suggestions(
            full_name,
            alumni_df,
            io_contacts_df
        )

        if full_name and not suggestion_df.empty:
            st.markdown("### Possible Matches in Current Data Pool")
            st.dataframe(
                suggestion_df,
                use_container_width=True,
                hide_index=True
            )

        elif full_name:
            st.info(
                "No exact person match found in the current data pool. "
                "The tool will continue searching by company overlap."
            )

    if "submitted" in locals() and submitted:

        if not full_name:
            st.warning("Please enter at least a full name.")

        else:
            result_df = search_new_person(
                full_name=full_name,
                current_company=current_company,
                previous_company=previous_company,
                alumni_df=alumni_df,
                io_contacts_df=io_contacts_df
            )

            st.divider()
            st.markdown("## Relationship Search Results")

            if result_df.empty:
                st.info(
                    "No Brown network relationship path found based on the current inputs."
                )

            else:
                total_score = min(result_df["score_contribution"].sum(), 100)

                if total_score >= 80:
                    overall_confidence = "High"
                elif total_score >= 50:
                    overall_confidence = "Medium"
                else:
                    overall_confidence = "Low"

                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

                summary_col1.metric("Target Person", full_name)
                summary_col2.metric("Total Score", int(total_score))
                summary_col3.metric("Overall Confidence", overall_confidence)
                summary_col4.metric("Evidence Count", len(result_df))

                st.progress(int(total_score))

                st.caption(
                    "Total score is the sum of all relationship evidence scores, capped at 100."
                )

                st.divider()

                st.markdown("### Relationship Evidence Details")

                for _, row in result_df.iterrows():
                    with st.container(border=True):
                        card_col1, card_col2 = st.columns([2, 1])

                        with card_col1:
                            st.markdown(f"#### {row['match_type']}")
                            st.write("**Target Person:**", row["target_person"])
                            st.write("**Brown-related Contact:**", row["brown_related_contact"])

                            if row["related_io_member"]:
                                st.write("**Related IO Member:**", row["related_io_member"])
                            else:
                                st.write("**Related IO Member:** N/A")

                            st.info(row["relationship_path"])

                        with card_col2:
                            st.write("**Confidence:**", row["confidence"])
                            st.write("**Score Contribution:**", int(row["score_contribution"]))
                            st.progress(min(int(row["score_contribution"]), 100))
                            st.write("**Evidence:**")
                            st.caption(row["evidence"])

                st.divider()

                st.markdown("### Ranked Evidence Table")

                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True
                )


# ============================================================
# 10. Data Sources Page
# ============================================================

elif current_page == "DataSources":

    st.markdown(
        """
        <div class="page-title">Data Sources</div>
        <div class="page-subtitle">
            Preview of the two data pools currently used in the prototype.
        </div>
        """,
        unsafe_allow_html=True
    )

    tab1, tab2 = st.tabs(["Brown Alumni Data", "Investment Office Contact Data"])

    with tab1:
        st.markdown("### Brown Alumni Data")
        st.write(
            "This table contains Brown alumni names, employers, job titles, "
            "constituency information, and managed relationship indicators."
        )

        st.dataframe(
            alumni_df,
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        st.markdown("### Investment Office Contact Data")
        st.write(
            "This table contains LinkedIn contact exports from Investment Office members, "
            "including contact names, companies, positions, and related IO members."
        )

        st.dataframe(
            io_contacts_df,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 11. Scoring Logic Page
# ============================================================

elif current_page == "ScoringLogic":

    st.markdown(
        """
        <div class="page-title">Scoring Logic</div>
        <div class="page-subtitle">
            Feature-weighted scoring system used to rank possible relationship paths.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(
        "The prototype uses a feature-weighted rule-based scoring system. "
        "The score does not evaluate the quality of a person or organization. "
        "It measures the relative strength of possible relationship paths."
    )

    st.divider()

    st.markdown("### Feature Weights")

    score_table = pd.DataFrame([
        {
            "Feature": "Direct IO LinkedIn Contact",
            "Weight": 40,
            "Meaning": "The person appears directly in an IO member's contact list."
        },
        {
            "Feature": "Current Employer Alumni Overlap",
            "Weight": 25,
            "Meaning": "The person's current company matches a Brown alum's current employer."
        },
        {
            "Feature": "Current Employer IO Contact Overlap",
            "Weight": 22,
            "Meaning": "The person's current company matches an IO contact's company."
        },
        {
            "Feature": "Previous Employer Alumni Overlap",
            "Weight": 18,
            "Meaning": "The person's previous company matches a Brown alum's current employer."
        },
        {
            "Feature": "Previous Employer IO Contact Overlap",
            "Weight": 15,
            "Meaning": "The person's previous company matches an IO contact's company."
        },
        {
            "Feature": "Managed Alumni Contact",
            "Weight": 10,
            "Meaning": "The matched Brown alum is marked as a managed relationship."
        }
    ])

    st.dataframe(
        score_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.markdown("### Score Formula")

    st.code(
        """
Total Score = sum of score contributions from all matched relationship evidence

Total Score is capped at 100.

Example:
Direct IO Contact = 40
Current Employer Alumni Overlap = 25
Current Employer IO Contact Overlap = 22

Total Score = 40 + 25 + 22 = 87
        """,
        language="text"
    )