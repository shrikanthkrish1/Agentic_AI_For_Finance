import pandas as pd
import numpy as np
import warnings
import traceback
from typing import Optional, Dict, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataCleaningTool:
    """Handles data cleaning and preparation for financial analysis"""
    
    def __init__(self):
        self.cleaning_log = []
    
    def clean_financial_data(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Clean and prepare financial data"""
        
        if raw_data is None or raw_data.empty:
            return {
                'success': False,
                'message': "No data provided for cleaning",
                'data': None,
                'log': []
            }
        
        try:
            df = raw_data.copy()
            self.cleaning_log = ["Starting data cleaning process..."]
            original_shape = df.shape
            
            # Log initial state
            self.cleaning_log.append(f"Original dataset: {original_shape[0]:,} rows x {original_shape[1]} columns")
            
            # Clean financial formatting from text columns
            for col in df.columns:
                if df[col].dtype == "object":
                    # Store original sample for logging
                    original_sample = df[col].head(3).tolist() if len(df) > 0 else []
                    
                    # Remove common financial formatting
                    df[col] = (df[col].astype(str)
                              .str.replace(",", "", regex=False)
                              .str.replace("%", "", regex=False)
                              .str.replace("$", "", regex=False)
                              .str.replace("€", "", regex=False)
                              .str.replace("£", "", regex=False)
                              .str.replace("₹", "", regex=False)
                              .str.replace("(", "-", regex=False)
                              .str.replace(")", "", regex=False)
                              .str.replace(" ", "", regex=False)
                              .str.replace("--", "-", regex=False)
                              .str.strip())
                    
                    # Try to convert to numeric
                    numeric_version = pd.to_numeric(df[col], errors='coerce')
                    non_null_count = numeric_version.notna().sum()
                    conversion_rate = non_null_count / len(df) if len(df) > 0 else 0
                    
                    if conversion_rate >= 0.7:  # Convert if 70%+ success rate
                        df[col] = numeric_version
                        self.cleaning_log.append(f"{col}: Converted to numeric ({conversion_rate:.1%} success)")
                        if len(original_sample) > 0:
                            self.cleaning_log.append(f"   Before: {original_sample}")
                            self.cleaning_log.append(f"   After: {df[col].head(3).tolist()}")
                    else:
                        self.cleaning_log.append(f"{col}: Kept as text ({conversion_rate:.1%} convertible)")
            
            # Handle missing values in numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) > 0:
                missing_before = df[numeric_cols].isnull().sum().sum()
                
                if missing_before > 0:
                    self.cleaning_log.append(f"Handling {missing_before} missing values...")
                    
                    for col in numeric_cols:
                        missing_count = df[col].isnull().sum()
                        if missing_count > 0:
                            median_value = df[col].median()
                            df[col] = df[col].fillna(median_value)
                            self.cleaning_log.append(f"{col}: Filled {missing_count} missing with median ({median_value:,.2f})")
            
            # Check for outliers
            quality_checks = []
            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0:
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                        
                        if outliers > 0:
                            quality_checks.append(f"{col}: {outliers} potential outliers detected")
            
            # Generate summary report
            final_shape = df.shape
            numeric_count = len(numeric_cols)
            
            report_lines = [
                "DATA CLEANING COMPLETED",
                "=" * 60,
                "TRANSFORMATION SUMMARY",
                "=" * 60,
                "",
                "Dataset Changes:",
                f"• Original: {original_shape[0]:,} rows x {original_shape[1]} columns",
                f"• Final: {final_shape[0]:,} rows x {final_shape[1]} columns",
                f"• Numeric columns: {numeric_count}",
                "",
                "Operations Performed:"
            ]
            
            for log_entry in self.cleaning_log:
                report_lines.append(f"  {log_entry}")
            
            report_lines.extend([
                "",
                "Data Quality Report:"
            ])
            
            if quality_checks:
                for check in quality_checks:
                    report_lines.append(f"  {check}")
            else:
                report_lines.append("  No major quality issues detected")
            
            report_lines.extend([
                "",
                "Status: Ready for Monte Carlo Simulation!",
                "=" * 60
            ])
            
            return {
                'success': True,
                'message': "\n".join(report_lines),
                'data': df,
                'log': self.cleaning_log,
                'numeric_columns': numeric_cols
            }
            
        except Exception as e:
            error_msg = f"Data cleaning failed: {str(e)}"
            return {
                'success': False,
                'message': error_msg,
                'data': None,
                'log': [error_msg]
            }


class MonteCarloTool:
    """Runs Monte Carlo simulations for financial analysis"""
    
    def __init__(self):
        self.simulation_results = None
        self.scenario_outcomes = {}
        self.stats = {}
    
    def run_simulation(self, cleaned_data: pd.DataFrame, n_simulations: int = 1000) -> Dict[str, Any]:
        """Execute Monte Carlo simulation"""
        
        if cleaned_data is None or cleaned_data.empty:
            return {
                'success': False,
                'message': "No cleaned data available for simulation",
                'results': None
            }
        
        try:
            df = cleaned_data.copy()
            
            # Map columns to financial categories
            column_mapping = self._identify_financial_columns(df)
            
            if len(column_mapping) < 2:
                return {
                    'success': False,
                    'message': f"""Insufficient financial data detected.
                    
Available columns: {list(df.columns)}
Detected financial columns: {list(column_mapping.keys())}

Your data should contain columns for revenue/sales, expenses/costs, and net income/profit.""",
                    'results': None
                }
            
            # Prepare financial data for simulation
            financial_data = df[[col for col in column_mapping.values()]].dropna()
            
            if len(financial_data) < 3:
                return {
                    'success': False,
                    'message': f"Insufficient data points. Found {len(financial_data)} complete rows, need at least 3.",
                    'results': None
                }
            
            # Rename columns and calculate derived metrics
            reverse_mapping = {v: k for k, v in column_mapping.items()}
            financial_data = financial_data.rename(columns=reverse_mapping)
            
            # Add cash flow if missing
            if 'Cash_Flow' not in financial_data.columns:
                if 'Net_Income' in financial_data.columns and 'Revenue' in financial_data.columns:
                    financial_data['Cash_Flow'] = (financial_data['Net_Income'] * 1.1 + 
                                                  financial_data['Revenue'] * 0.05)
            
            # Calculate key ratios
            revenue_col = financial_data['Revenue']
            expenses_col = financial_data.get('Expenses', revenue_col * 0.7)
            net_income_col = financial_data.get('Net_Income', revenue_col - expenses_col)
            
            financial_data['expense_ratio'] = np.where(revenue_col != 0, expenses_col / revenue_col, 0.7)
            financial_data['profit_margin'] = np.where(revenue_col != 0, net_income_col / revenue_col, 0.05)
            
            # Clip ratios to realistic business ranges
            financial_data['expense_ratio'] = financial_data['expense_ratio'].clip(0.2, 2.0)
            financial_data['profit_margin'] = financial_data['profit_margin'].clip(-0.3, 0.5)
            
            # Calculate statistical parameters for simulation
            ratio_cols = ['expense_ratio', 'profit_margin']
            ratios_df = financial_data[ratio_cols].dropna()
            
            if len(ratios_df) == 0:
                return {
                    'success': False,
                    'message': "Unable to calculate financial ratios from the data.",
                    'results': None
                }
            
            mean_ratios = ratios_df.mean()
            cov_matrix = ratios_df.cov()
            
            # Model revenue growth
            revenue_series = financial_data['Revenue'].dropna()
            growth_params = self._calculate_growth_parameters(revenue_series)
            
            last_revenue = revenue_series.iloc[-1]
            
            # Run Monte Carlo simulation
            simulation_data = []
            scenario_counts = {"excellent": 0, "good": 0, "neutral": 0, "poor": 0, "crisis": 0}
            
            successful_sims = 0
            for i in range(n_simulations):
                try:
                    # Simulate revenue growth with multiple factors
                    market_shock = np.random.normal(0, 0.05)
                    company_growth = np.random.normal(growth_params['mu'], growth_params['sigma'])
                    trend_influence = growth_params['recent_trend'] * 0.2
                    
                    total_growth = company_growth + market_shock + trend_influence
                    total_growth = np.clip(total_growth, -0.6, 0.8)
                    
                    sim_revenue = last_revenue * (1 + total_growth)
                    
                    # Simulate correlated financial ratios
                    try:
                        sim_ratios = np.random.multivariate_normal(mean_ratios.values, cov_matrix.values)
                        expense_ratio, profit_margin = sim_ratios
                    except:
                        expense_ratio = np.random.normal(mean_ratios['expense_ratio'], ratios_df['expense_ratio'].std())
                        profit_margin = np.random.normal(mean_ratios['profit_margin'], ratios_df['profit_margin'].std())
                    
                    # Apply business constraints
                    expense_ratio = np.clip(expense_ratio, 0.3, 1.8)
                    profit_margin = np.clip(profit_margin, -0.25, 0.4)
                    
                    # Calculate financial results
                    sim_expenses = sim_revenue * expense_ratio
                    sim_net_income = sim_revenue * profit_margin
                    sim_cash_flow = sim_net_income * 1.1
                    
                    roi = profit_margin
                    
                    # Classify scenarios
                    if roi > 0.20:
                        scenario_counts["excellent"] += 1
                    elif roi > 0.10:
                        scenario_counts["good"] += 1
                    elif roi >= 0:
                        scenario_counts["neutral"] += 1
                    elif roi >= -0.10:
                        scenario_counts["poor"] += 1
                    else:
                        scenario_counts["crisis"] += 1
                    
                    simulation_data.append([
                        sim_revenue, sim_expenses, sim_cash_flow, sim_net_income, roi
                    ])
                    successful_sims += 1
                    
                except Exception:
                    continue
            
            if len(simulation_data) == 0:
                return {
                    'success': False,
                    'message': "Monte Carlo simulation failed to generate results.",
                    'results': None
                }
            
            # Create results dataframe
            sim_df = pd.DataFrame(
                simulation_data,
                columns=['Revenue', 'Expenses', 'Cash_Flow', 'Net_Income', 'ROI']
            )
            
            # Calculate comprehensive statistics
            self.stats = self._calculate_simulation_stats(sim_df)
            self.scenario_outcomes = scenario_counts
            self.simulation_results = sim_df
            
            # Generate report
            report = self._generate_simulation_report(successful_sims)
            
            return {
                'success': True,
                'message': report,
                'results': sim_df,
                'stats': self.stats,
                'scenarios': scenario_counts
            }
            
        except Exception as e:
            error_msg = f"Monte Carlo simulation failed: {str(e)}"
            return {
                'success': False,
                'message': error_msg,
                'results': None
            }
    
    def _identify_financial_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Identify financial columns in the dataframe"""
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['revenue', 'sales', 'income', 'turnover']):
                column_mapping['Revenue'] = col
            elif any(keyword in col_lower for keyword in ['expense', 'cost', 'expenditure']):
                column_mapping['Expenses'] = col
            elif any(keyword in col_lower for keyword in ['net', 'profit', 'earning']):
                column_mapping['Net_Income'] = col
        
        return column_mapping
    
    def _calculate_growth_parameters(self, revenue_series: pd.Series) -> Dict[str, float]:
        """Calculate revenue growth parameters"""
        if len(revenue_series) > 1:
            growth_rates = revenue_series.pct_change().dropna()
            if len(growth_rates) > 0:
                mu_growth = growth_rates.mean()
                sigma_growth = growth_rates.std()
                recent_trend = growth_rates.tail(3).mean()
            else:
                mu_growth, sigma_growth, recent_trend = 0.05, 0.10, 0.05
        else:
            mu_growth, sigma_growth, recent_trend = 0.05, 0.10, 0.05
        
        # Cap extreme values
        mu_growth = np.clip(mu_growth, -0.3, 0.5)
        sigma_growth = np.clip(sigma_growth, 0.02, 0.3)
        
        return {
            'mu': mu_growth,
            'sigma': sigma_growth,
            'recent_trend': recent_trend
        }
    
    def _calculate_simulation_stats(self, sim_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive simulation statistics"""
        ni = sim_df['Net_Income']
        revenue = sim_df['Revenue']
        roi = sim_df['ROI']
        
        stats = {
            'ni_mean': ni.mean(),
            'ni_median': ni.median(),
            'ni_std': ni.std(),
            'revenue_mean': revenue.mean(),
            'revenue_std': revenue.std(),
            'roi_mean': roi.mean(),
            'prob_loss': (ni < 0).mean() * 100,
            'prob_profit': (ni > 0).mean() * 100,
            'prob_high_return': (roi > 0.15).mean() * 100,
            'var_5': ni.quantile(0.05),
            'var_95': ni.quantile(0.95)
        }
        
        # Calculate conditional value at risk
        var_5 = stats['var_5']
        stats['cvar_5'] = ni[ni <= var_5].mean() if (ni <= var_5).any() else var_5
        
        return stats
    
    def _generate_simulation_report(self, successful_sims: int) -> str:
        """Generate comprehensive simulation report"""
        stats = self.stats
        scenario_counts = self.scenario_outcomes
        
        report_lines = [
            "MONTE CARLO SIMULATION COMPLETED!",
            f"({successful_sims:,} successful simulations)",
            "",
            "=" * 70,
            "COMPREHENSIVE FINANCIAL PROJECTIONS",
            "=" * 70,
            "",
            "NET INCOME ANALYSIS:",
            f"• Expected Value: ${stats['ni_mean']:,.0f}",
            f"• Median Outcome: ${stats['ni_median']:,.0f}",
            f"• Standard Deviation: ${stats['ni_std']:,.0f}",
            f"• Range: ${self.simulation_results['Net_Income'].min():,.0f} to ${self.simulation_results['Net_Income'].max():,.0f}",
            "",
            "REVENUE PROJECTIONS:",
            f"• Expected Revenue: ${stats['revenue_mean']:,.0f}",
            f"• Revenue Volatility: ${stats['revenue_std']:,.0f}",
            f"• Revenue Range: ${self.simulation_results['Revenue'].min():,.0f} to ${self.simulation_results['Revenue'].max():,.0f}",
            "",
            "PROFITABILITY ANALYSIS:",
            f"• Expected ROI: {stats['roi_mean']:.2%}",
            f"• Probability of Profit: {stats['prob_profit']:.1f}%",
            f"• Probability of Loss: {stats['prob_loss']:.1f}%",
            f"• High Return Probability (>15%): {stats['prob_high_return']:.1f}%",
            "",
            "RISK ASSESSMENT:",
            f"• Value at Risk (5%): ${stats['var_5']:,.0f}",
            f"• Expected Shortfall: ${stats['cvar_5']:,.0f}",
            f"• Upside Potential (95%): ${stats['var_95']:,.0f}",
            "",
            "SCENARIO DISTRIBUTION:"
        ]
        
        for scenario, count in scenario_counts.items():
            percentage = count / 10
            emoji = {"excellent": "🌟", "good": "🎯", "neutral": "⚖️", "poor": "⚠️", "crisis": "🚨"}
            report_lines.append(f"• {emoji.get(scenario, '•')} {scenario.title()} (>20% ROI): {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "=" * 70,
            "SIMULATION COMPLETE - READY FOR STORYTELLING!",
            "=" * 70
        ])
        
        return "\n".join(report_lines)


class FinancialStoryteller:
    """Creates engaging financial narratives from simulation results"""
    
    def __init__(self):
        self.story_templates = {
            'excellent': "golden scenarios",
            'good': "success stories", 
            'neutral': "balanced realities",
            'poor': "testing grounds",
            'crisis': "crisis chronicles"
        }
    
    def create_story(self, simulation_results: pd.DataFrame, scenario_outcomes: Dict[str, int], 
                     stats: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive financial story"""
        
        if simulation_results is None or len(simulation_results) == 0:
            return {
                'success': False,
                'message': "No simulation results available for storytelling",
                'story': None
            }
        
        try:
            story_content = self._build_narrative(scenario_outcomes, stats)
            
            return {
                'success': True,
                'message': "Financial story generated successfully",
                'story': story_content
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Storytelling error: {str(e)}",
                'story': None
            }
    
    def _build_narrative(self, scenario_outcomes: Dict[str, int], stats: Dict[str, float]) -> str:
        """Build the complete financial narrative"""
        
        story_sections = []
        
        # Prologue
        story_sections.append(self._create_prologue())
        
        # Scenario chapters
        for scenario_type in ['excellent', 'good', 'neutral', 'poor', 'crisis']:
            count = scenario_outcomes.get(scenario_type, 0)
            percentage = count / 10
            section = self._create_scenario_chapter(scenario_type, count, percentage)
            story_sections.append(section)
        
        # Grand revelation
        story_sections.append(self._create_revelation(stats))
        
        # Strategic wisdom
        story_sections.append(self._create_strategic_wisdom(stats, scenario_outcomes))
        
        # Epilogue
        story_sections.append(self._create_epilogue())
        
        return "\n\n".join(story_sections)
    
    def _create_prologue(self) -> str:
        return """THE FINANCIAL CRYSTAL BALL: Your Business Future Unveiled

PROLOGUE

Imagine holding a magical crystal ball that reveals 1,000 different futures for your business. Each future is a possible timeline, shaped by market forces, strategic decisions, and the inherent uncertainties of business. Let me take you on a journey through these alternate realities..."""
    
    def _create_scenario_chapter(self, scenario_type: str, count: int, percentage: float) -> str:
        """Create a chapter for each scenario type"""
        
        chapters = {
            'excellent': f"""CHAPTER 1: THE GOLDEN SCENARIOS
*The Realm of Excellence*

In {count} out of 1,000 futures ({percentage:.1f}%), your business becomes a legendary success story!

The Excellence Narrative:
Picture this: You wake up to quarterly reports that make you smile. Your profit margins soar above 20%, making your business a beacon of efficiency and market dominance. These are the timelines where your strategic vision aligns perfectly with market opportunities.

In these golden scenarios, your revenue streams flow like mighty rivers, your cost management is poetry in motion, and every decision you made in the past pays dividends. Competitors study your business model, investors queue at your door, and industry publications feature your success story.

Expected Performance: ROI > 20%, representing the top tier of business excellence""",

            'good': f"""CHAPTER 2: THE SUCCESS STORIES
*The Land of Solid Growth*

In {count} parallel universes ({percentage:.1f}%), your business achieves commendable success!

The Growth Narrative:
These are the futures where you build a sustainable, profitable enterprise that consistently outperforms market averages. With profit margins dancing between 10-20%, you're in the sweet spot of business success.

Imagine steady quarterly growth, loyal customers singing your praises, a motivated team that believes in your vision, and the financial stability to weather minor storms while planning for bigger opportunities. This is your business operating like a well-oiled machine, not necessarily breaking speed records, but reaching its destination reliably and profitably.

Expected Performance: ROI 10-20%, the foundation of lasting business success""",

            'neutral': f"""CHAPTER 3: THE BALANCED REALITIES
*The Middle Kingdom*

In {count} scenarios ({percentage:.1f}%), your business finds itself in the middle ground of modest success.

The Stability Narrative:
These futures represent the steady heartbeat of business survival and gradual growth. You're generating modest profits (0-10% margins), building experience, and establishing market presence. While not spectacular, these scenarios represent resilience and learning.

Think of these as your business's training montage - each challenge overcome, each lesson learned, each small victory building toward something greater. You're not just surviving; you're evolving, adapting, and positioning yourself for future breakthroughs.

Expected Performance: ROI 0-10%, the stepping stones to greater success""",

            'poor': f"""CHAPTER 4: THE TESTING GROUNDS
*The Valley of Challenges*

In {count} timelines ({percentage:.1f}%), your business faces headwinds that test your resolve.

The Resilience Narrative:
These are the scenarios where the market throws curveballs, competition intensifies, or unexpected costs emerge. Your margins slip into slightly negative territory (-10% to 0%), but these aren't death sentences - they're character-building experiences.

Imagine navigating through a storm - challenging, requiring careful resource management and strategic pivots, but with clear pathways back to calmer waters. These scenarios reveal the importance of emergency funds, adaptability, and strong leadership during tough times.

Expected Performance: ROI -10% to 0%, temporary setbacks with recovery potential""",

            'crisis': f"""CHAPTER 5: THE CRISIS CHRONICLES
*The Crucible of Transformation*

In {count} futures ({percentage:.1f}%), your business faces its greatest trials.

The Phoenix Narrative:
These scenarios involve significant challenges where losses exceed 10% of revenue. Market crashes, major disruptions, supply chain failures, or significant operational crises create perfect storms.

But here's the twist in our story: even these scenarios aren't the end. They're transformation catalysts. Like a phoenix rising from ashes, businesses that survive these trials often emerge stronger, more resilient, and better positioned for future success. These scenarios underscore the critical importance of contingency planning and crisis management.

Expected Performance: ROI < -10%, severe challenges requiring strategic transformation"""
        }
        
        return chapters.get(scenario_type, "")
    
    def _create_revelation(self, stats: Dict[str, float]) -> str:
        return f"""THE GRAND REVELATION

Your Destiny Awaits:
Looking across all 1,000 possible futures, the crystal ball reveals your expected net income of ${stats['ni_mean']:,.0f}. This isn't just a number - it's the mathematical poetry of probability, the weighted wisdom of uncertainty.

The Probability Prophecy:
• {stats['prob_profit']:.1f}% chance of profitable futures
• {stats['prob_loss']:.1f}% chance of challenging times
• In the darkest 5% of scenarios: ${abs(stats['var_5']):,.0f} potential loss
• In the brightest 5% of scenarios: ${stats['var_95']:,.0f} potential gain"""
    
    def _create_strategic_wisdom(self, stats: Dict[str, float], scenario_outcomes: Dict[str, int]) -> str:
        risk_level = "conservative" if stats['prob_loss'] < 25 else "balanced" if stats['prob_loss'] < 45 else "aggressive"
        risk_description = "a relatively safe harbor" if stats['prob_loss'] < 30 else "dynamic waters where higher risks meet higher rewards" if stats['prob_loss'] < 50 else "adventurous seas with significant upside potential"
        
        diversification = "excellent scenario diversification" if max(scenario_outcomes.values()) < 400 else "concentrated outcome patterns"
        diversification_desc = "a well-balanced risk-reward portfolio with multiple paths to success" if max(scenario_outcomes.values()) < 400 else "potential for both significant breakthroughs and notable challenges"
        
        return f"""THE STRATEGIC WISDOM

Your Risk-Reward Profile:
The simulation reveals your business operates in a {risk_level} risk environment. With a {stats['prob_loss']:.1f}% probability of losses, you're positioned in {risk_description}.

The Scenario Spectrum Analysis:
Your business demonstrates {diversification}, suggesting {diversification_desc}."""
    
    def _create_epilogue(self) -> str:
        return """EPILOGUE: THE STRATEGIC COMPASS

This isn't just a story - it's your strategic intelligence transformed into narrative wisdom. Your roadmap forward:

Maximize Excellence: Focus on the factors that increase the probability of those golden 20%+ ROI scenarios
Prepare for Storms: Develop robust contingency plans for the crisis scenarios
Amplify Success: Build systems and processes that make good outcomes more likely
Stabilize the Foundation: Convert neutral scenarios into launching pads for growth
Mitigate Risks: Implement early warning systems and rapid response protocols

THE FINAL WISDOM: The future isn't written in stone - these scenarios show you the landscape of possibilities. Your strategic decisions, market timing, and execution excellence will determine which path your business actually takes!

Remember: Every great business story started with someone who dared to navigate uncertainty with wisdom, courage, and strategic intelligence.

YOUR STORY IS WAITING TO BE WRITTEN!"""
