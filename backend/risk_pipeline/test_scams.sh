#!/bin/bash
HOST="http://127.0.0.1:8000"

# Colors for better readability
GREEN='\033[032m'
RED='\033[031m'
NC='\033[0m' # No Color

test_case() {
    title=$1
    text=$2
    echo -e "${GREEN}Testing: $title${NC}"
    echo "Text: \"$text\""
    response=$(curl -s -X POST $HOST/v1/classify -H 'Content-Type: application/json' -d "{\"text\": \"$text\"}")
    
    label=$(echo $response | grep -oP '"label":"\K[^"]+')
    score=$(echo $response | grep -oP '"risk_score":\K[0-9.]+')
    heuristic=$(echo $response | grep -oP '"heuristic_triggered":\K[^,]+')
    
    echo -e "Result: ${RED}$label${NC} (Score: $score, Heuristic: $heuristic)"
    echo "------------------------------------------------"
}

test_case "The Doubler" "Send 0.05 BTC to this address and get 0.1 BTC back instantly! Verified giveaway!"
test_case "The Insider" "Exclusive insider info! Join my VIP group for 50x monthly gains. Only 2 spots left!"
test_case "The Airdrop" "FREE AIRDROP! Claim your \$10,000 Pepe tokens now before the timer hits zero: [link]"
test_case "The Guaranteed Daily" "Revolutionary AI bot guarantees 5% daily profit with zero drawdown. Start with \$100 today."
test_case "The Secret Strategy" "I made \$50k in one week using this secret strategy. DM me CASH to learn how for free!"
