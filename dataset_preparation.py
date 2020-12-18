import sys


def main():
    try:
        data = open(sys.argv[1])
        with open('new_'+sys.argv[1], 'w') as new_data:
            new_data.write('id,di,fi1,fi2,fi3,fi4,fi5,fi6,fi7,fi8,fi9,fi10,fi11' \
                           ',fi12,fi13,fi14,fi15,fi16,fi17,fi18,fi19,fi20,fi21,fi22' \
                           ',fi23,fi24,fi25,fi26,fi27,fi28,fi29,fi30\n')
            for line in data:
                new_data.write(line)
    except:
        print('Please add data and try again')


if __name__ == '__main__':
    main()
