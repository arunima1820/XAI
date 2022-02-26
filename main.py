from foldrpp import *
from timeit import default_timer as timer
from datetime import timedelta

# https://www.kaggle.com/namanmanchanda/entrepreneurial-competency-in-university-students


def entrepreneur():
    attrs = ["EducationSector", "IndividualProject", "Age", "Gender", "Influenced", "Perseverance", "DesireToTakeInitiative",
             "Competitiveness", "SelfReliance", "StrongNeedToAchieve", "SelfConfidence", "GoodPhysicalHealth", "KeyTraits"]
    nums = ['Age']

    model = Classifier(attrs=attrs, numeric=nums, label='y', pos='0')
    data = model.load_data('data/Entrepreneur.csv')
    print('\n% Entrepreneur dataset', len(data), len(data[0]))
    return model, data

# https://www.kaggle.com/tejashvi14/employee-future-prediction


def employeeRetention():
    attrs = ['Education', 'JoiningYear', 'PaymentTier',
             'Age', 'Gender', 'ExperienceInCurrentDomain']

    nums = ['Age']

    model = Classifier(attrs=attrs, numeric=nums, label='LeaveOrNot', pos='0')
    data = model.load_data('data/Employee.csv')
    print('\n% Employee dataset', len(data), len(data[0]))
    return model, data


# https://www.kaggle.com/pritsheta/diabetes-dataset
def diabetes():
    attrs = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
             'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    nums = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    model = Classifier(attrs=attrs, numeric=nums, label='Outcome', pos='1')
    data = model.load_data('data/diabetes.csv')
    print('\n% Diabetes dataset', len(data), len(data[0]))
    return model, data


def main():
    model, data = entrepreneur()
    data_train, data_test = split_data(data, ratio=0.8, rand=True)

    X_train, Y_train = split_xy(data_train)
    X_test,  Y_test = split_xy(data_test)

    start = timer()
    model.fit(X_train, Y_train, ratio=0.8)
    end = timer()

    save_model_to_file(model, 'employeeRetention.model')
    # model.print_asp(simple=True)

    Y_test_hat = model.predict(X_test)
    acc, p, r, f1 = get_scores(Y_test_hat, Y_test)
    print('% acc', round(acc, 4), 'p', round(p, 4),
          'r', round(r, 4), 'f1', round(f1, 4))
    print('% foldr++ costs: ', timedelta(seconds=end - start), '\n')

    model2 = load_model_from_file('employeeRetention.model')
    model2.print_asp(simple=True)

    # k = 1
    # for i in range(10):
    #     print('Explanation for example number', k, ':')
    #     print(model.explain(X_test[i], all_flag=False))
    #     print('Proof Trees for example number', k, ':')
    #     print(model.proof(X_test[i], all_flag=False))
    #     k += 1


if __name__ == '__main__':
    main()
