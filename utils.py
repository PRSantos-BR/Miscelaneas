from datetime import datetime


def inicio_processo() -> datetime:
    dataHoraInicioProcesso: datetime = datetime.now()
    print('-- Início do processo.', '\n', dataHoraInicioProcesso.strftime('%d/%m/%Y %H:%M:%S'), '\n')

    print('processando ...', '\n \n')

    return dataHoraInicioProcesso


def fim_processo(dataHoraInicioProcesso: datetime) -> None:
    print('\n\n', '... processado.', '\n')

    # Data e hora do "FIM do processo.
    dataHoraFimProcesso: datetime = datetime.now()
    print('-- Fim do processo.', '\n', dataHoraFimProcesso.strftime('%d/%m/%Y %H:%M:%S'), '\n')
    print('-- Processo executado em : ', dataHoraFimProcesso - dataHoraInicioProcesso)



