import gc

import numpy as np
import pandas as pd


def drop_duplicates_pharma(df):
    """
    df: long-format dataframe of a patient
    varref: variable reference table that contain the mean and standard deviation of values for a subset of variables
    """
    df_dup = df[df.duplicated(["givenat", "pharmaid", "infusionid"], keep=False)]
    for pharmaid in df_dup.pharmaid.unique():
        for infusionid in df_dup[df_dup.pharmaid == pharmaid].infusionid.unique():
            tmp = df_dup[(df_dup.pharmaid == pharmaid) & (df_dup.infusionid == infusionid)]
            if len(tmp.recordstatus.unique()) == 1 and tmp.recordstatus.unique()[0] == 780:
                for i in range(len(tmp)):
                    df.loc[tmp.index[i], "infusionid"] = "%s_%s" % (int(df.loc[tmp.index[i], "infusionid"]), i)
                # tmp = df[(df.pharmaid == pharmaid) & (
                #     df.infusionid.apply(lambda x: "%s_" % (infusionid) in x if type(x) == str else False))]
            elif len(tmp.recordstatus.unique()) == 1 and tmp.recordstatus.unique()[0] == 776:
                if (tmp.givendose != 0).sum() == 1:
                    df.drop(tmp.index[tmp.givendose == 0], inplace=True)
                else:
                    df.drop(tmp.index[:-1], inplace=True)
            elif len(tmp.recordstatus.unique()) == 2 and 776 in tmp.recordstatus.unique():
                df.drop(tmp.index[tmp.recordstatus != 776], inplace=True)
            else:
                raise Exception("Debug needed")
    return df


def process_status780(df, acting_period):
    '''
    Convert the infusion channel with status injection/tablet to "infusion-like" channel.
    '''
    infusionid = int(df.iloc[0].infusionid)
    start_code = 524
    stop_code = 776

    df.set_index("givenat", inplace=True)
    drug_giventime_780 = df.index.tolist()

    df_new = []
    for i, dt in enumerate(drug_giventime_780):
        tmp = df.loc[[dt]].copy()

        endtime_780 = dt + np.timedelta64(acting_period, "m")
        tmp.loc[endtime_780, "givendose"] = tmp.loc[dt, "givendose"]
        tmp.loc[endtime_780, "recordstatus"] = stop_code
        tmp.loc[endtime_780, "infusionid"] = "%d_%d" % (infusionid, i)

        tmp.loc[dt, "givendose"] = 0
        tmp.loc[dt, "recordstatus"] = start_code
        tmp.loc[dt, "infusionid"] = "%d_%d" % (infusionid, i)

        df_new.append(tmp.reset_index())
    df_new = pd.concat(df_new).sort_values("givenat")
    return df_new


def process_single_infusion(df, acting_period):
    '''
    Convert given dose from a single infusion channel to rate
    '''
    infusionid = int(df.iloc[0].infusionid)
    if len(df.recordstatus.unique()) == 1 and df.recordstatus.unique()[0] == 780:
        df = process_status780(df, acting_period)

    df_rate = []
    for sub_infusionid in df.infusionid.unique():
        tmp = df[df.infusionid == sub_infusionid].copy()
        try:
            assert ((tmp.recordstatus == 524).sum() == 1)
        except AssertionError:
            tmp.set_index("givenat", inplace=True)
            beg_time = tmp.index[0] - np.timedelta64(acting_period, "m")
            tmp.loc[beg_time, "givendose"] = 0
            tmp.loc[beg_time, "recordstatus"] = 524
            tmp.loc[beg_time, "infusionid"] = sub_infusionid
            tmp.sort_index(inplace=True)
            tmp.reset_index(inplace=True)
        try:
            assert ((tmp.recordstatus == 776).sum() == 1)
        except AssertionError:
            pass
        tmp.loc[:, "rate"] = 0
        tmp.loc[tmp.index[:-1], "rate"] = tmp.givendose.values[1:] / (tmp.givenat.diff() / np.timedelta64(1,
                                                                                                          "m")).values[
                                                                     1:]
        tmp.rename(columns={"rate": str(sub_infusionid)}, inplace=True)
        df_rate.append(tmp[["givenat", str(sub_infusionid)]].set_index("givenat"))
    df_rate = pd.concat(df_rate, axis=1).sum(axis=1).to_frame(name=str(infusionid))
    return df_rate


def convert_cumul_value_to_rate(df, cumul_urine_id_lst, general_table):
    pid = df.iloc[0].patientid
    short_gap = 5 / 60

    rec_adm_time = general_table.loc[pid].admissiontime
    # if the first HR measuremet time is earlier than recorded admission time, then we estimated
    # the "true" admission time to be the earlier of these two time points.
    if df[df.variableid == 200]["value"].notnull().sum() > 0:
        hr_first_meas_time = df.loc[df[df.variableid == 200]["value"].notnull().index[0], "datetime"]
        esti_adm_time = min(rec_adm_time, hr_first_meas_time)
    else:
        esti_adm_time = rec_adm_time

    df_urine = df[df.variableid.isin(cumul_urine_id_lst)]

    if len(df_urine) == 0:
        return df
    else:
        for vid in df_urine.variableid.unique():
            df_tmp = df_urine[df_urine.variableid == vid]  # table of a single urine variable

            index_pre_general_table = df_tmp.index[df_tmp.datetime < esti_adm_time - np.timedelta64(15 * 60 + 30,
                                                                                                    "s")]  # number of records before general_admission time
            if len(index_pre_general_table) == 0:
                pass
            elif len(index_pre_general_table) == 1:
                # if there's one record before general_admission, reset datetime from system reset time 12pm to the general_admission time
                index_pre_general_table = df_tmp.index[df_tmp.datetime < esti_adm_time]
                df.loc[index_pre_general_table[0], 'datetime'] = esti_adm_time
            else:
                index_pre_general_table = df_tmp.index[df_tmp.datetime < esti_adm_time]
                df.drop(index_pre_general_table[:-1], inplace=True)
                df.loc[index_pre_general_table[-1], 'datetime'] = esti_adm_time

            df_tmp = df[df.variableid == vid]
            if df_tmp.duplicated(["datetime"]).sum() == 0:
                pass
            else:
                df.drop(df_tmp.index[df_tmp.duplicated(["datetime"])], inplace=True)

            # delete urine record if therre's only one left
            if (df.variableid == vid).sum() < 2:
                df.drop(df.index[df.variableid == vid], inplace=True)
                continue

            # compute the cumulative values over the entire icu stay
            df_tmp = df[df.variableid == vid]
            t_reset = df_tmp[(df_tmp["value"].diff() < 0) | (
                    df_tmp.index == df_tmp.index[0])].datetime  # reset time for the cumulative counting
            idx_not_reset = df_tmp[(df_tmp["value"].diff() >= 0) & (df_tmp.index != df_tmp.index[0])].index
            for i in np.arange(1, len(t_reset)):
                tmp = df_tmp[df_tmp.datetime >= t_reset.iloc[i]]
                if i < len(t_reset) - 1:
                    tmp = tmp[tmp.datetime < t_reset.iloc[i + 1]]
                df.loc[tmp.index, 'value'] += df.loc[df_tmp.index[df_tmp.datetime < t_reset.iloc[i]][-1], 'value']

            # drop the time point with time difference from the previous time point that is smaller than half an hour
            df_tmp = df[df.variableid == vid]
            tdiff = (df_tmp.datetime.diff().iloc[1:] / np.timedelta64(3600, 's'))
            if (tdiff < short_gap).sum() > 0:
                df.drop(df_tmp.index[1:][tdiff.values < short_gap], inplace=True)

            if (df.variableid == vid).sum() < 2:
                df.drop(df.index[df.variableid == vid], inplace=True)
                continue

            # debug if the cumulative value is not strictly increasing
            df_tmp = df[df.variableid == vid]
            vdiff = df_tmp["value"].diff()
            try:
                assert ((vdiff < 0).sum() == 0)
            except AssertionError:
                import ipdb
                ipdb.set_trace()
        gc.collect()

        for vid in df_urine.variableid.unique():
            df_tmp = df[df.variableid == vid]
            if len(df_tmp) == 0:
                continue
            elif len(df_tmp) == 1:
                continue
            else:
                tdiff = (df_tmp.datetime.diff() / np.timedelta64(3600, 's'))
                df.loc[df_tmp.index[1:], 'value'] = (df_tmp["value"].diff().iloc[1:] / tdiff.iloc[1:]).values
                #                 logging.info(tdiff.loc[df.loc[df_tmp.index,'value']>1e+4]*np.timedelta64(3600, 's'))
                #             df.loc[df_tmp.index[0],'value'] = df.loc[df_tmp.index[1],'value']
                df.loc[df_tmp.index[0], 'value'] = 0

        for vid in df_urine.variableid.unique():
            df_tmp = df[df.variableid == vid]
        #             df.drop(df_tmp.index[df_tmp['value'] > 1e+6], inplace=True)

        return df
