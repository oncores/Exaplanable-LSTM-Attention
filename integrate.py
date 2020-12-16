import preprocessing.paragraph_analysis
import pickle
import main.sentence_analysis

data = "./data/인수계약서_111번.xlsx"

#문단,문장의 시각화 결과 출력 여부 on으로 parameter 설정시 각각의 문단과 문장 시각화 출력 
#off로 설정시 시각화 결과를 추출하지 않고 문단과 문장 분류만 실행
visualize='on'
# visualize='off'

prob_pkl, num2label = preprocessing.paragraph_analysis.main(data, visualize)
main.sentence_analysis.main(prob_pkl, num2label, data, visualize)