import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_damodaran_industry_betas():
    """
    Fetch industry median levered betas from Damodaran's NYU Stern data.
    Returns a DataFrame with industry names and their corresponding betas.
    """
    url = "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/Betas.html"
    
    try:
        # Fetch the webpage
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table containing beta data
        tables = soup.find_all('table')
        
        if not tables:
            st.error("No tables found on the Damodaran beta page")
            return pd.DataFrame()
        
        # Try to find the correct table (usually the first or second table)
        beta_data = []
        
        for table in tables:
            rows = table.find_all('tr')
            if len(rows) < 5:  # Skip small tables
                continue
                
            headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]
            
            # Look for columns that might contain industry and beta data
            industry_col = None
            beta_col = None
            
            for i, header in enumerate(headers):
                if 'industry' in header.lower() or 'sector' in header.lower():
                    industry_col = i
                if 'beta' in header.lower() and ('levered' in header.lower() or 'unlevered' not in header.lower()):
                    beta_col = i
                    break
            
            # If we found relevant columns, extract data
            if industry_col is not None and beta_col is not None:
                for row in rows[1:]:  # Skip header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) > max(industry_col, beta_col):
                        industry = cells[industry_col].get_text().strip()
                        beta_text = cells[beta_col].get_text().strip()
                        
                        # Extract numeric beta value
                        beta_match = re.search(r'(\d+\.?\d*)', beta_text)
                        if beta_match and industry and industry != 'Industry Name':
                            try:
                                beta_value = float(beta_match.group(1))
                                beta_data.append({
                                    'Industry': industry,
                                    'Levered_Beta': beta_value
                                })
                            except ValueError:
                                continue
                
                # If we found data, break from table loop
                if beta_data:
                    break
        
        if not beta_data:
            # Fallback: try to parse any table with beta-like data
            for table in tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip potential header
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        industry = cells[0].get_text().strip()
                        beta_text = cells[1].get_text().strip()
                        
                        # Look for beta values
                        beta_match = re.search(r'(\d+\.?\d*)', beta_text)
                        if beta_match and industry and len(industry) > 3:
                            try:
                                beta_value = float(beta_match.group(1))
                                if 0.1 <= beta_value <= 3.0:  # Reasonable beta range
                                    beta_data.append({
                                        'Industry': industry,
                                        'Levered_Beta': beta_value
                                    })
                            except ValueError:
                                continue
                
                if beta_data:
                    break
        
        if beta_data:
            df = pd.DataFrame(beta_data)
            # Clean up industry names
            df['Industry'] = df['Industry'].str.replace(r'^\d+\.?\s*', '', regex=True)  # Remove leading numbers
            df['Industry'] = df['Industry'].str.strip()
            
            # Remove duplicates and sort
            df = df.drop_duplicates(subset=['Industry']).sort_values('Industry')
            
            return df
        else:
            st.warning("Could not parse beta data from Damodaran website")
            return pd.DataFrame()
            
    except requests.RequestException as e:
        st.error(f"Failed to fetch Damodaran beta data: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error parsing Damodaran beta data: {e}")
        return pd.DataFrame()

def find_industry_beta(industry_betas_df: pd.DataFrame, company_sector: str, company_industry: str = None):
    """
    Find the most relevant industry beta for a company.
    
    Args:
        industry_betas_df: DataFrame with industry beta data
        company_sector: Company's sector
        company_industry: Company's specific industry (optional)
    
    Returns:
        Dict with matched industry name and beta value, or None if not found
    """
    if industry_betas_df.empty:
        return None
    
    # Create search terms from sector and industry
    search_terms = []
    if company_industry:
        search_terms.append(company_industry.lower())
    if company_sector:
        search_terms.append(company_sector.lower())
    
    # Try exact matches first
    for term in search_terms:
        exact_match = industry_betas_df[
            industry_betas_df['Industry'].str.lower() == term
        ]
        if not exact_match.empty:
            return {
                'industry': exact_match.iloc[0]['Industry'],
                'beta': exact_match.iloc[0]['Levered_Beta']
            }
    
    # Try partial matches
    for term in search_terms:
        partial_matches = industry_betas_df[
            industry_betas_df['Industry'].str.lower().str.contains(term, na=False)
        ]
        if not partial_matches.empty:
            return {
                'industry': partial_matches.iloc[0]['Industry'],
                'beta': partial_matches.iloc[0]['Levered_Beta']
            }
    
    # Try broader keyword matching
    keywords = []
    for term in search_terms:
        keywords.extend(term.split())
    
    for keyword in keywords:
        if len(keyword) > 3:  # Skip very short words
            keyword_matches = industry_betas_df[
                industry_betas_df['Industry'].str.lower().str.contains(keyword, na=False)
            ]
            if not keyword_matches.empty:
                return {
                    'industry': keyword_matches.iloc[0]['Industry'],
                    'beta': keyword_matches.iloc[0]['Levered_Beta']
                }
    
    return None

def calculate_wacc_with_industry_beta(market_beta: float, industry_beta: float = None, 
                                    risk_free_rate: float = 0.03, market_premium: float = 0.06,
                                    debt_ratio: float = 0.3, cost_of_debt: float = 0.05, 
                                    tax_rate: float = 0.25):
    """
    Calculate WACC using both company-specific beta and industry benchmark beta.
    
    Returns:
        Dict with WACC calculations for both company and industry betas
    """
    results = {}
    
    # Company WACC using company-specific beta
    cost_of_equity_company = risk_free_rate + (market_beta * market_premium)
    wacc_company = ((1 - debt_ratio) * cost_of_equity_company) + (debt_ratio * cost_of_debt * (1 - tax_rate))
    
    results['company_wacc'] = {
        'wacc': wacc_company,
        'cost_of_equity': cost_of_equity_company,
        'beta': market_beta
    }
    
    # Industry WACC using industry benchmark beta
    if industry_beta:
        cost_of_equity_industry = risk_free_rate + (industry_beta * market_premium)
        wacc_industry = ((1 - debt_ratio) * cost_of_equity_industry) + (debt_ratio * cost_of_debt * (1 - tax_rate))
        
        results['industry_wacc'] = {
            'wacc': wacc_industry,
            'cost_of_equity': cost_of_equity_industry,
            'beta': industry_beta
        }
    
    return results