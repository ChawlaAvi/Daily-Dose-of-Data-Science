#!/usr/bin/env python3
"""
Hotel Guests Data Analysis
==========================

This script analyzes hotel guest data to extract meaningful insights and create visualizations.
The analysis includes:
1. Room type distribution and revenue analysis
2. Rewards program impact on spending
3. Seasonal booking patterns
4. Geographic distribution of guests
5. Amenities fee analysis
6. Length of stay patterns

Author: Codegen AI Assistant
Date: June 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_clean_data(filepath):
    """Load and clean the hotel guests dataset."""
    print("📊 Loading and cleaning hotel guests data...")
    
    # Load the data
    df = pd.read_csv(filepath)
    
    # Basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Handle missing values
    print(f"\nMissing values per column:")
    print(df.isnull().sum())
    
    # Convert date columns
    df['checkin_date'] = pd.to_datetime(df['checkin_date'], format='%d %b %Y', errors='coerce')
    df['checkout_date'] = pd.to_datetime(df['checkout_date'], format='%d %b %Y', errors='coerce')
    
    # Calculate length of stay
    df['length_of_stay'] = (df['checkout_date'] - df['checkin_date']).dt.days
    
    # Extract month and season from check-in date
    df['checkin_month'] = df['checkin_date'].dt.month
    df['checkin_season'] = df['checkin_month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })
    
    # Extract state from billing address
    df['state'] = df['billing_address'].str.extract(r', ([A-Z]{2}) \d{5}')
    
    # Calculate total cost (room rate + amenities fee)
    df['total_cost'] = df['room_rate'] + df['amenities_fee'].fillna(0)
    
    print(f"\n✅ Data cleaning completed!")
    print(f"Valid check-in dates: {df['checkin_date'].notna().sum()}")
    print(f"Valid length of stay calculations: {df['length_of_stay'].notna().sum()}")
    
    return df

def create_room_type_analysis(df):
    """Analyze room types and their revenue contribution."""
    print("\n🏨 Analyzing room types and revenue...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Room Type Analysis', fontsize=16, fontweight='bold')
    
    # Room type distribution
    room_counts = df['room_type'].value_counts()
    ax1.pie(room_counts.values, labels=room_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribution of Room Types')
    
    # Average room rate by type
    avg_rates = df.groupby('room_type')['room_rate'].mean().sort_values(ascending=True)
    ax2.barh(avg_rates.index, avg_rates.values)
    ax2.set_title('Average Room Rate by Type')
    ax2.set_xlabel('Average Room Rate ($)')
    
    # Total revenue by room type
    total_revenue = df.groupby('room_type')['total_cost'].sum().sort_values(ascending=True)
    ax3.barh(total_revenue.index, total_revenue.values)
    ax3.set_title('Total Revenue by Room Type')
    ax3.set_xlabel('Total Revenue ($)')
    
    # Room rate distribution by type
    df.boxplot(column='room_rate', by='room_type', ax=ax4)
    ax4.set_title('Room Rate Distribution by Type')
    ax4.set_xlabel('Room Type')
    ax4.set_ylabel('Room Rate ($)')
    
    plt.tight_layout()
    plt.savefig('room_type_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return room_counts, avg_rates, total_revenue

def analyze_rewards_program(df):
    """Analyze the impact of rewards program on guest behavior."""
    print("\n🎁 Analyzing rewards program impact...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Rewards Program Analysis', fontsize=16, fontweight='bold')
    
    # Rewards program distribution
    rewards_dist = df['has_rewards'].value_counts()
    ax1.pie(rewards_dist.values, labels=['No Rewards', 'Has Rewards'], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Rewards Program Participation')
    
    # Average spending by rewards status
    avg_spending = df.groupby('has_rewards')['total_cost'].mean()
    ax2.bar(['No Rewards', 'Has Rewards'], avg_spending.values)
    ax2.set_title('Average Total Spending by Rewards Status')
    ax2.set_ylabel('Average Total Cost ($)')
    
    # Amenities fee by rewards status
    amenities_by_rewards = df.groupby('has_rewards')['amenities_fee'].mean()
    ax3.bar(['No Rewards', 'Has Rewards'], amenities_by_rewards.values)
    ax3.set_title('Average Amenities Fee by Rewards Status')
    ax3.set_ylabel('Average Amenities Fee ($)')
    
    # Room type preference by rewards status
    room_rewards = pd.crosstab(df['room_type'], df['has_rewards'], normalize='columns') * 100
    room_rewards.plot(kind='bar', ax=ax4)
    ax4.set_title('Room Type Preference by Rewards Status (%)')
    ax4.set_ylabel('Percentage')
    ax4.legend(['No Rewards', 'Has Rewards'])
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('rewards_program_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return rewards_dist, avg_spending, amenities_by_rewards

def analyze_seasonal_patterns(df):
    """Analyze seasonal booking patterns."""
    print("\n🗓️ Analyzing seasonal booking patterns...")
    
    # Filter out rows with missing check-in dates
    df_valid_dates = df.dropna(subset=['checkin_date'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Seasonal Booking Patterns', fontsize=16, fontweight='bold')
    
    # Monthly booking distribution
    monthly_bookings = df_valid_dates['checkin_month'].value_counts().sort_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax1.bar(range(1, 13), [monthly_bookings.get(i, 0) for i in range(1, 13)])
    ax1.set_title('Bookings by Month')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Bookings')
    ax1.set_xticks(range(1, 13))
    ax1.set_xticklabels(month_names, rotation=45)
    
    # Seasonal distribution
    seasonal_bookings = df_valid_dates['checkin_season'].value_counts()
    ax2.pie(seasonal_bookings.values, labels=seasonal_bookings.index, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Bookings by Season')
    
    # Average room rate by season
    seasonal_rates = df_valid_dates.groupby('checkin_season')['room_rate'].mean()
    ax3.bar(seasonal_rates.index, seasonal_rates.values)
    ax3.set_title('Average Room Rate by Season')
    ax3.set_ylabel('Average Room Rate ($)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Length of stay by season
    df_valid_dates.boxplot(column='length_of_stay', by='checkin_season', ax=ax4)
    ax4.set_title('Length of Stay by Season')
    ax4.set_xlabel('Season')
    ax4.set_ylabel('Length of Stay (days)')
    
    plt.tight_layout()
    plt.savefig('seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return monthly_bookings, seasonal_bookings, seasonal_rates

def analyze_geographic_distribution(df):
    """Analyze geographic distribution of guests."""
    print("\n🗺️ Analyzing geographic distribution...")
    
    # Filter out rows with valid state information
    df_with_states = df.dropna(subset=['state'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Geographic Distribution of Guests', fontsize=16, fontweight='bold')
    
    # Top states by number of guests
    top_states = df_with_states['state'].value_counts().head(10)
    ax1.barh(range(len(top_states)), top_states.values)
    ax1.set_yticks(range(len(top_states)))
    ax1.set_yticklabels(top_states.index)
    ax1.set_title('Top 10 States by Number of Guests')
    ax1.set_xlabel('Number of Guests')
    
    # Average spending by top states
    state_spending = df_with_states.groupby('state')['total_cost'].mean().sort_values(ascending=False).head(10)
    ax2.barh(range(len(state_spending)), state_spending.values)
    ax2.set_yticks(range(len(state_spending)))
    ax2.set_yticklabels(state_spending.index)
    ax2.set_title('Top 10 States by Average Spending')
    ax2.set_xlabel('Average Total Cost ($)')
    
    plt.tight_layout()
    plt.savefig('geographic_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return top_states, state_spending

def analyze_amenities_and_stay_patterns(df):
    """Analyze amenities fees and length of stay patterns."""
    print("\n🛎️ Analyzing amenities and stay patterns...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Amenities and Stay Patterns Analysis', fontsize=16, fontweight='bold')
    
    # Amenities fee distribution
    amenities_with_fee = df[df['amenities_fee'] > 0]['amenities_fee']
    ax1.hist(amenities_with_fee, bins=20, edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Amenities Fees (Excluding $0)')
    ax1.set_xlabel('Amenities Fee ($)')
    ax1.set_ylabel('Frequency')
    
    # Amenities fee vs room rate correlation
    df_clean = df.dropna(subset=['amenities_fee', 'room_rate'])
    ax2.scatter(df_clean['room_rate'], df_clean['amenities_fee'], alpha=0.6)
    ax2.set_title('Amenities Fee vs Room Rate')
    ax2.set_xlabel('Room Rate ($)')
    ax2.set_ylabel('Amenities Fee ($)')
    
    # Length of stay distribution
    stay_lengths = df.dropna(subset=['length_of_stay'])['length_of_stay']
    stay_lengths = stay_lengths[stay_lengths > 0]  # Remove invalid stays
    ax3.hist(stay_lengths, bins=range(1, int(stay_lengths.max()) + 2), edgecolor='black', alpha=0.7)
    ax3.set_title('Distribution of Length of Stay')
    ax3.set_xlabel('Length of Stay (days)')
    ax3.set_ylabel('Frequency')
    
    # Average total cost by length of stay
    stay_cost = df.groupby('length_of_stay')['total_cost'].mean()
    stay_cost = stay_cost[stay_cost.index > 0]  # Remove invalid stays
    ax4.bar(stay_cost.index, stay_cost.values)
    ax4.set_title('Average Total Cost by Length of Stay')
    ax4.set_xlabel('Length of Stay (days)')
    ax4.set_ylabel('Average Total Cost ($)')
    
    plt.tight_layout()
    plt.savefig('amenities_stay_patterns.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return amenities_with_fee, stay_lengths, stay_cost

def generate_summary_report(df):
    """Generate a comprehensive summary report."""
    print("\n📋 Generating summary report...")
    
    # Calculate key metrics
    total_guests = len(df)
    total_revenue = df['total_cost'].sum()
    avg_room_rate = df['room_rate'].mean()
    avg_amenities_fee = df['amenities_fee'].mean()
    rewards_participation = (df['has_rewards'].sum() / total_guests) * 100
    avg_length_of_stay = df['length_of_stay'].mean()
    
    # Most popular room type
    most_popular_room = df['room_type'].mode()[0]
    
    # Peak booking month
    peak_month = df.dropna(subset=['checkin_month'])['checkin_month'].mode()
    peak_month_name = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 
                       5: 'May', 6: 'June', 7: 'July', 8: 'August', 
                       9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    peak_month_str = peak_month_name.get(peak_month[0], 'Unknown') if len(peak_month) > 0 else 'Unknown'
    
    report = f"""
    ═══════════════════════════════════════════════════════════════
                        HOTEL GUESTS ANALYSIS REPORT
    ═══════════════════════════════════════════════════════════════
    
    📊 OVERVIEW METRICS
    ─────────────────────────────────────────────────────────────
    • Total Guests: {total_guests:,}
    • Total Revenue: ${total_revenue:,.2f}
    • Average Room Rate: ${avg_room_rate:.2f}
    • Average Amenities Fee: ${avg_amenities_fee:.2f}
    • Average Length of Stay: {avg_length_of_stay:.1f} days
    
    🎯 KEY INSIGHTS
    ─────────────────────────────────────────────────────────────
    • Rewards Program Participation: {rewards_participation:.1f}%
    • Most Popular Room Type: {most_popular_room}
    • Peak Booking Month: {peak_month_str}
    
    💡 BUSINESS RECOMMENDATIONS
    ─────────────────────────────────────────────────────────────
    1. REWARDS PROGRAM: {rewards_participation:.1f}% participation rate suggests 
       opportunity to increase enrollment through targeted marketing.
    
    2. ROOM TYPE OPTIMIZATION: {most_popular_room} rooms are most popular - 
       consider inventory allocation and pricing strategies.
    
    3. SEASONAL PLANNING: {peak_month_str} shows highest booking volume - 
       optimize staffing and pricing for peak periods.
    
    4. AMENITIES STRATEGY: Average amenities fee of ${avg_amenities_fee:.2f} 
       indicates potential for revenue optimization through package deals.
    
    5. LENGTH OF STAY: Average stay of {avg_length_of_stay:.1f} days suggests 
       opportunities for extended stay packages and loyalty incentives.
    
    ═══════════════════════════════════════════════════════════════
    """
    
    print(report)
    
    # Save report to file
    with open('analysis_report.txt', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main analysis function."""
    print("🏨 Hotel Guests Data Analysis Starting...")
    print("=" * 60)
    
    # Load and clean data
    df = load_and_clean_data('guests.csv')
    
    # Perform analyses
    room_analysis = create_room_type_analysis(df)
    rewards_analysis = analyze_rewards_program(df)
    seasonal_analysis = analyze_seasonal_patterns(df)
    geographic_analysis = analyze_geographic_distribution(df)
    amenities_analysis = analyze_amenities_and_stay_patterns(df)
    
    # Generate summary report
    summary_report = generate_summary_report(df)
    
    print("\n✅ Analysis completed successfully!")
    print("📁 Generated files:")
    print("   • room_type_analysis.png")
    print("   • rewards_program_analysis.png") 
    print("   • seasonal_patterns.png")
    print("   • geographic_distribution.png")
    print("   • amenities_stay_patterns.png")
    print("   • analysis_report.txt")
    print("\n🎉 All visualizations and reports have been saved!")

if __name__ == "__main__":
    main()
