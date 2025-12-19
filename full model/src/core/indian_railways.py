class IndianWagonParser:
    """
    Parses 11-digit Indian Railways Wagon Numbers.
    Format:
    C1 C2       : Wagon Type
    C3 C4       : Owning Railway
    C5 C6       : Year of Manufacture
    C7 C8 C9 C10: Unique ID
    C11         : Check Digit
    """
    
    WAGON_TYPES = {
        '10': 'BOXN', '11': 'BOXNHA', '12': 'BOXNHS', '13': 'BOXNCR', '14': 'BOXNLW',
        '15': 'BOXNB', '16': 'BOXNF', '17': 'BOXNG', '18': 'BOY', '19': 'BOST',
        '20': 'BOXNAL', '21': 'BOXN-HS', '22': 'BOXNHL', '24': 'BOXNS',
        '30': 'BCN', '31': 'BCNA', '32': 'BCNAHS', '40': 'BTPN', '41': 'BTPGLN',
        '42': 'BTALN', '43': 'BTCS', '44': 'BTPH', '45': 'BTAP', '46': 'BTFLN', 
        # Add more mappings as needed
    }

    RAILWAY_CODES = {
        '01': 'CR', '02': 'ER', '03': 'NR', '04': 'NER', '05': 'NFR', '06': 'SR',
        '07': 'SER', '08': 'WR', '09': 'SCR', '16': 'NWR', '17': 'SWR', '11': 'ECR',
        '12': 'ECoR', '13': 'NCR', '14': 'SECR', '10': 'EC', '15': 'WCR', '26': 'Metro'
    }

    @staticmethod
    def parse(number_str):
        # Clean input: remove spaces, non-digits
        clean_num = ''.join(filter(str.isdigit, number_str))
        
        if len(clean_num) != 11:
            return None # Not a valid 11-digit code

        c1_c2 = clean_num[0:2]
        c3_c4 = clean_num[2:4]
        c5_c6 = clean_num[4:6]
        c7_c10 = clean_num[6:10]
        c11 = clean_num[10]

        wagon_type = IndianWagonParser.WAGON_TYPES.get(c1_c2, "Unknown")
        railway = IndianWagonParser.RAILWAY_CODES.get(c3_c4, "Unknown")
        
        # Year logic: 00-99. Assuming 2000+? Or 1900? 
        # Usually contextual, but let's just return the raw YY
        year_mfg = f"20{c5_c6}" # Approximation suitable for modern wagons
        
        # Check Digit Validation (Standard approach involves specific weights, 
        # but for now we just parse. We can implement Luhn or specific algo if specs provided)
        
        return {
            "original": clean_num,
            "formatted": f"{c1_c2} {c3_c4} {c5_c6} {c7_c10} {c11}",
            "type": wagon_type,
            "railway": railway,
            "year": year_mfg,
            "id": c7_c10,
            "check_digit": c11
        }

    @staticmethod
    def validate_checksum(number_str):
        # Placeholder for specific checksum algo
        return True
