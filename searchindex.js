Search.setIndex({docnames:["index","res/code/interface","res/code/model_framework","res/code/model_instance_seg","res/code/model_workers","res/code/plugin_base","res/code/plugin_convert","res/code/plugin_crop","res/code/plugin_metrics","res/code/plugin_model_inference","res/code/plugin_model_training","res/code/plugin_review","res/code/plugin_review_dock","res/code/utils","res/guides/cropping_module_guide","res/guides/custom_model_template","res/guides/detailed_walkthrough","res/guides/inference_module_guide","res/guides/metrics_module_guide","res/guides/review_module_guide","res/guides/training_module_guide","res/guides/utils_module_guide","res/welcome"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":5,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,sphinx:56},filenames:["index.rst","res/code/interface.rst","res/code/model_framework.rst","res/code/model_instance_seg.rst","res/code/model_workers.rst","res/code/plugin_base.rst","res/code/plugin_convert.rst","res/code/plugin_crop.rst","res/code/plugin_metrics.rst","res/code/plugin_model_inference.rst","res/code/plugin_model_training.rst","res/code/plugin_review.rst","res/code/plugin_review_dock.rst","res/code/utils.rst","res/guides/cropping_module_guide.rst","res/guides/custom_model_template.rst","res/guides/detailed_walkthrough.rst","res/guides/inference_module_guide.rst","res/guides/metrics_module_guide.rst","res/guides/review_module_guide.rst","res/guides/training_module_guide.rst","res/guides/utils_module_guide.rst","res/welcome.rst"],objects:{"napari_cellseg3d.code_models.model_framework":[[2,0,1,"","ModelFramework"]],"napari_cellseg3d.code_models.model_framework.ModelFramework":[[2,1,1,"","_viewer"],[2,1,1,"","docked_widgets"],[2,1,1,"","images_filepaths"],[2,1,1,"","labels_filepaths"],[2,1,1,"","results_path"],[2,1,1,"","worker"]],"napari_cellseg3d.code_models.model_instance_seg":[[3,2,1,"","binary_connected"],[3,2,1,"","binary_watershed"],[3,2,1,"","clear_small_objects"],[3,2,1,"","to_instance"],[3,2,1,"","to_semantic"],[3,2,1,"","volume_stats"]],"napari_cellseg3d.code_plugins.plugin_base":[[5,0,1,"","BasePluginFolder"],[5,0,1,"","BasePluginSingleImage"]],"napari_cellseg3d.code_plugins.plugin_base.BasePluginFolder":[[5,1,1,"","_viewer"],[5,1,1,"","images_filepaths"],[5,1,1,"","labels_filepaths"],[5,1,1,"","results_path"]],"napari_cellseg3d.code_plugins.plugin_base.BasePluginSingleImage":[[5,1,1,"","_viewer"],[5,1,1,"","image_layer_loader"],[5,1,1,"","image_path"],[5,1,1,"","label_layer_loader"],[5,1,1,"","label_path"]],"napari_cellseg3d.code_plugins.plugin_convert":[[6,0,1,"","AnisoUtils"],[6,0,1,"","InstanceWidgets"],[6,0,1,"","RemoveSmallUtils"],[6,0,1,"","ThresholdUtils"],[6,0,1,"","ToInstanceUtils"],[6,0,1,"","ToSemanticUtils"],[6,2,1,"","save_folder"],[6,2,1,"","save_layer"],[6,2,1,"","show_result"]],"napari_cellseg3d.code_plugins.plugin_convert.AnisoUtils":[[6,3,1,"","__init__"]],"napari_cellseg3d.code_plugins.plugin_convert.InstanceWidgets":[[6,3,1,"","__init__"],[6,3,1,"","run_method"]],"napari_cellseg3d.code_plugins.plugin_convert.RemoveSmallUtils":[[6,3,1,"","__init__"]],"napari_cellseg3d.code_plugins.plugin_convert.ThresholdUtils":[[6,3,1,"","__init__"]],"napari_cellseg3d.code_plugins.plugin_convert.ToInstanceUtils":[[6,3,1,"","__init__"]],"napari_cellseg3d.code_plugins.plugin_convert.ToSemanticUtils":[[6,3,1,"","__init__"]],"napari_cellseg3d.code_plugins.plugin_crop":[[7,0,1,"","Cropping"]],"napari_cellseg3d.code_plugins.plugin_crop.Cropping":[[7,1,1,"","_viewer"],[7,1,1,"","image_path"],[7,1,1,"","label_path"]],"napari_cellseg3d.code_plugins.plugin_metrics":[[8,0,1,"","MetricsUtils"]],"napari_cellseg3d.code_plugins.plugin_metrics.MetricsUtils":[[8,1,1,"","_viewer"],[8,1,1,"","canvas"],[8,3,1,"","layout"],[8,1,1,"","plots"]],"napari_cellseg3d.code_plugins.plugin_model_inference":[[9,0,1,"","Inferer"]],"napari_cellseg3d.code_plugins.plugin_model_inference.Inferer":[[9,1,1,"","_viewer"],[9,1,1,"","config"],[9,1,1,"","instance_config"],[9,1,1,"","model_info"],[9,1,1,"","post_process_config"],[9,1,1,"","worker"],[9,1,1,"","worker_config"]],"napari_cellseg3d.code_plugins.plugin_model_training":[[10,0,1,"","Trainer"]],"napari_cellseg3d.code_plugins.plugin_model_training.Trainer":[[10,1,1,"","_viewer"],[10,1,1,"","canvas"],[10,1,1,"","dice_metric_plot"],[10,1,1,"","loss_dict"],[10,1,1,"","train_loss_plot"],[10,1,1,"","worker"]],"napari_cellseg3d.code_plugins.plugin_review":[[11,0,1,"","Reviewer"]],"napari_cellseg3d.code_plugins.plugin_review.Reviewer":[[11,1,1,"","_viewer"],[11,1,1,"","image_path"],[11,1,1,"","label_path"]],"napari_cellseg3d.code_plugins.plugin_review_dock":[[12,0,1,"","Datamanager"]],"napari_cellseg3d.code_plugins.plugin_review_dock.Datamanager":[[12,1,1,"","viewer"]],"napari_cellseg3d.interface":[[1,0,1,"","AnisotropyWidgets"],[1,0,1,"","Button"],[1,0,1,"","CheckBox"],[1,0,1,"","ContainerWidget"],[1,0,1,"","DoubleIncrementCounter"],[1,0,1,"","DropdownMenu"],[1,0,1,"","FilePathWidget"],[1,0,1,"","IntIncrementCounter"],[1,0,1,"","Log"],[1,0,1,"","QWidgetSingleton"],[1,0,1,"","ScrollArea"],[1,0,1,"","UtilsDropdown"],[1,2,1,"","add_blank"],[1,2,1,"","add_label"],[1,2,1,"","combine_blocks"],[1,2,1,"","handle_adjust_errors"],[1,2,1,"","handle_adjust_errors_wrapper"],[1,2,1,"","make_group"],[1,2,1,"","open_file_dialog"],[1,2,1,"","open_url"],[1,2,1,"","toggle_visibility"]],"napari_cellseg3d.interface.AnisotropyWidgets":[[1,3,1,"","__init__"],[1,3,1,"","build"],[1,3,1,"","enabled"],[1,3,1,"","resolution_xyz"],[1,3,1,"","resolution_zyx"],[1,3,1,"","scaling_xyz"],[1,3,1,"","scaling_zyx"]],"napari_cellseg3d.interface.Button":[[1,3,1,"","__init__"],[1,3,1,"","visibility_condition"]],"napari_cellseg3d.interface.CheckBox":[[1,3,1,"","__init__"]],"napari_cellseg3d.interface.ContainerWidget":[[1,3,1,"","__init__"]],"napari_cellseg3d.interface.DoubleIncrementCounter":[[1,3,1,"","__init__"]],"napari_cellseg3d.interface.DropdownMenu":[[1,3,1,"","__init__"]],"napari_cellseg3d.interface.FilePathWidget":[[1,3,1,"","__init__"],[1,3,1,"","build"],[1,4,1,"","button"],[1,3,1,"","check_ready"],[1,4,1,"","text_field"],[1,3,1,"","update_field_color"]],"napari_cellseg3d.interface.IntIncrementCounter":[[1,3,1,"","__init__"]],"napari_cellseg3d.interface.Log":[[1,3,1,"","__init__"],[1,3,1,"","print_and_log"],[1,3,1,"","replace_last_line"],[1,3,1,"","warn"],[1,3,1,"","write"]],"napari_cellseg3d.interface.QWidgetSingleton":[[1,3,1,"","__call__"]],"napari_cellseg3d.interface.ScrollArea":[[1,3,1,"","__init__"],[1,3,1,"","make_scrollable"]],"napari_cellseg3d.interface.UtilsDropdown":[[1,3,1,"","__init__"],[1,3,1,"","dropdown_menu_call"],[1,3,1,"","show_utils_menu"]],"napari_cellseg3d.utils":[[13,0,1,"","Singleton"],[13,2,1,"","denormalize_y"],[13,2,1,"","dice_coeff"],[13,2,1,"","format_Warning"],[13,2,1,"","get_date_time"],[13,2,1,"","get_padding_dim"],[13,2,1,"","get_time"],[13,2,1,"","get_time_filepath"],[13,2,1,"","load_images"],[13,2,1,"","normalize_x"],[13,2,1,"","normalize_y"],[13,2,1,"","save_stack"],[13,2,1,"","sphericity_axis"],[13,2,1,"","sphericity_volume_area"],[13,2,1,"","time_difference"]]},objnames:{"0":["py","class","Python class"],"1":["py","attribute","Python attribute"],"2":["py","function","Python function"],"3":["py","method","Python method"],"4":["py","property","Python property"]},objtypes:{"0":"py:class","1":"py:attribute","2":"py:function","3":"py:method","4":"py:property"},terms:{"0":[1,3,8,9,12,13,16,17,18,20,21,22],"019":22,"0554":22,"1":[1,3,9,13,16,18,21,22],"10":[1,3,9,16,22],"1038":22,"1073":22,"11":1,"120":[16,20],"128":[3,16,17,20],"1918465117":22,"1e":16,"2":[4,10,13,16,18,22],"20":[1,16],"2019":22,"2020":22,"255":13,"2d":[1,7,13,19,20],"3":[1,3,13,16,22],"30x40x100":13,"32x64x128":13,"3d":[1,7,13,16,17,19,20,22],"3dunet":[17,20],"40":[16,20],"5":[3,8,16,20],"6":13,"60":[16,20],"64":[16,20],"65":20,"7":1,"8":3,"9":3,"90":16,"98":3,"case":[13,16,19],"catch":[9,10],"class":[12,15],"default":[1,3,5,10,16,17],"do":[16,18,20],"final":[16,20],"float":[1,3],"function":[2,4,10,16,20],"import":[1,15,16],"int":[1,3,10,12,13],"long":[9,22],"new":[1,11,12,19],"return":[1,2,3,4,5,6,9,10,11,12,13,15],"short":16,"static":[2,4],"switch":16,"true":[1,2,3,4,5,9,10,12,13,20],"try":[13,16,20],"while":[10,16],A:[1,2,4,5,6,7,9,10,11,12,16,17,19,20,22],And:17,By:[16,17],For:[1,13,16,17,22],If:[1,2,7,9,10,11,13,14,16,17,18,19,20,22],In:[13,16,17],It:[18,19,22],NOT:13,Not:16,On:17,The:[1,3,9,16,17,18,20,22],Then:[16,22],There:16,These:22,To:[1,14,15,16,22],Will:[4,9],With:16,_:17,__call__:1,__init__:[1,2,4,5,6,7,8,9,10,11,12],_method:6,_pred:17,_start:7,_viewer:[2,5,7,8,9,10,11],a0:12,a_p:13,ab:13,about:22,abov:[1,3,14,16,17,19,20,22],acceler:16,accept:13,access:[1,9,13,14,16,19,22],accur:20,achard:22,action:21,activ:5,ad:[1,5,6,16],adapt:22,add:[1,2,6,7,11,13,15],add_dock_widget:2,addition:16,advanc:[16,22],after:[10,16,17,18,20],afterward:[9,17,21],again:[14,16,20],ah:12,al:22,algorithm:[3,17,22],all:[3,4,5,8,9,11,13,16,17,20,21],alloc:[3,10],allow:[1,11,12,14,17,18,19,20,22],alreadi:[9,16,18,19],also:[4,14,16,17,20,22],alwai:[13,19,20],always_vis:1,amount:4,amout:4,an:[1,2,3,4,6,9,13,16,17,18,20,21],analysi:0,analyz:16,ani:[2,16,18,20,22],anisotrop:[9,14,16,17,19,21],anisotropi:[1,6,14,17,19,21],anisotropy_factor:13,annot:[16,19],anoth:[1,4,14],anytim:16,anywher:19,appen:17,appli:17,applic:17,approach:16,appropri:[16,20],ar:[7,9,10,11,13,14,16,17,18,19,20,22],archiv:16,area:[1,13],arg:[1,3,12],argument:[1,4],around:[11,14],arrai:[1,2,3,5,6,8,13],artifact:[3,16],as_fold:[12,13],as_str:13,ask:[9,17,19],assign:16,assist:22,associ:[2,16],assum:[16,18],attempt:1,attribut:15,augment:[4,10,16,20],auto:16,autoencod:[17,20],automat:[2,10,16,17,18,20,22],autosav:20,avail:[2,16,17,20,22],avoid:[1,5,10,16,17],aw:12,ax:[3,12,13],axi:[1,9],axon:[16,22],ay:12,b:[1,13],back:16,background:1,bar:[1,2,9,17],base:[1,5,6,16,19],base_wh:1,basic:5,batch:[4,10,15,16,20],batch_siz:4,been:[1,9,11,13,16,17,19,20,22],befor:[1,11,16],beforehand:14,begin:[17,20],behaviour:13,being:[13,18,19,20],below:[1,9,16,17,18,20,21,22],beneath:17,best:[10,16,18],better:[4,10,11,14,16,17,19,20],between:[1,8,9,13,16,18],binar:17,binari:3,bind:1,blank:1,blue:18,bool:[1,4,12,13],both:[1,2,17,20],bottom:1,box:[1,9,10],brain:[16,17,20,22],broadli:17,broken:20,browser:1,build:[1,16],button:[2,5,6,7,9,10,11,14,16,17,18,19,20],c:[3,15],cach:2,calcul:16,call:[1,2,3,6,8,9,13,17],callabl:1,can:[1,2,5,7,9,11,13,14,16,17,18,19,20,21,22],cannot:[2,14],canva:[8,10],cap:[9,18],capabl:22,categori:13,caus:[16,20],ce:20,cell:[4,16,17,20,22],cellseg3d:[16,22],cellseg:20,center:[16,22],centroid:[3,16,17],cgi:22,chang:[14,16,17,18],channel:[10,15],check:[1,9,10,11,12,14,16,17,18,19],check_image_data:11,check_readi:[1,9,10],check_warn:13,checkbox:[9,11,12,14,19],children:[5,7,9,11],choic:[1,2,7,9,10,11,16,17],choos:[1,2,7,9,10,11,14,16,17,19,20,22],chose:[16,20],chosen:[1,4,6,7,9,10,11,17],chunk:17,classmethod:1,clean:9,clear:[8,10,16,22],click:[1,5,10,11,16,19,20],close:[1,7,9,11,16,20],cnn:[13,16],code:[13,16],code_model:[2,3,4,9],code_plugin:[5,6,7,8,9,10,11,12],coeffici:[13,18],collabor:22,collect:16,color:[1,16],colormap:[14,17],combin:1,come:[16,22],compar:16,comparison:18,compat:[13,16,17,20],compens:[16,18],complet:[16,18,20],compon:[3,16,17],compris:20,comput:[3,4,8,9,13,16,17,18,22],compute_dic:8,confid:17,config:[4,9],configur:4,confirm:11,connect:[1,2,3,16,17],consecut:1,consid:[3,16,17,18],consist:16,constant:14,contain:[1,2,3,4,6,9,13,16,17,19,20],contained_layout:1,content:1,context:1,contrast:[14,17],contribut:[16,22],control:[1,7,14,22],convers:[0,14,19,22],convert:[3,6,17,21,22],convolut:[17,20,22],coordin:[3,16,17],copi:[16,19,20],correct:[1,6,14,16,19,20,22],correctli:[1,9,10],correspond:[2,11,14,16,19],cortic:16,cost:17,could:16,counter:[1,16],cpu:[2,4,16,17],creat:[1,2,4,5,6,7,8,9,10,11,12,19],create_inference_dict:[4,9],create_train_dataset_dict:[2,4],credit:22,crop:[0,22],csv:[4,11,12,16,17,19,22],csv_cell_plot:16,ctrl:[1,5,14,16],cube:20,cubic:[16,20],cuda:[2,4,17,20,22],current:[2,3,8,12,15,17,19,20,21],custom:[0,4,9,10,16,17,20,22],cyril:22,d:[13,14,15,16,17],dask:13,data:[1,2,3,4,6,9,10,11,16,17,19,20,22],data_dict:4,data_path:10,dataclass:4,datafram:12,dataset:[2,4,10,11,12,13,16,17,19,20],date:[13,17],datetim:13,de:13,decent:16,decid:1,declar:[0,16,22],decreas:[16,20],deep:16,def:15,default_i:1,default_x:1,default_z:1,defin:[1,5,6,10,14,18,20,22],deform:[10,20],degrad:16,depend:[1,4,7,13,14,20,22],depth:15,describ:[3,14],descript:1,design:16,desir:[10,14,16],destroyoldwindow:12,detail:[0,22],detect:[0,22],determin:[1,4,6,11,18],determinist:[4,10,16,20],develop:22,devic:[2,4,17,20],dialog:[1,5,19],dice:[4,8,10,13,16,18,20],dice_coeff:8,dice_metr:10,dice_metric_plot:10,dict:[2,3,4,9,10],dictionari:[2,4],did:16,diff:13,differ:[13,16,21],dim:[9,13],dimens:[1,7,13],dir_or_path:13,direct:22,directli:[1,3,10,14,16,22],directori:[1,7,11,13,16,21],disabl:20,discov:2,disk:2,displai:[1,2,5,7,8,9,10,11,12,16,17,18,20],display_status_report:[2,9],dissimilar:16,do_augment:4,doc:[10,16,22],dock:[2,5,9,11],docked_widget:2,doe:[3,9,10,21],doi:22,done:[11,14,16,17],down:16,dr:22,dropdown:[1,5,6,7,9,10,11,14],dropdown_menu_cal:1,due:[1,16,18],dure:[10,16],dynam:[10,14],e:[2,3,4,9,11,13,14,16,17,22],each:[1,3,8,9,13,16,17,18,19,20,21,22],easier:[14,16,19,20,22],easili:17,edg:16,either:[2,13,16,17,19,20],elast:[10,20],element:[1,2,4],emit:2,empti:[1,2,10],empty_cuda_cach:2,enabl:[1,5,9,14,16,20],enable_utils_menu:5,end:[16,19],enough:3,ensur:[1,16,17,18,20],enter:[1,16],entir:[11,16],entri:1,epoch:[4,10,16,20],equal:[16,20],error:[1,9,10,22],especi:22,establish:19,et:22,etc:[2,10,20],evalu:[8,9,20],even:[1,16],event:1,everi:[4,9,16,20],exampl:[15,16,17,20,21],example1:1,example2:1,execut:1,exist:[1,19],expect:13,explain:16,extens:[1,2,4,5,6,7,9,10,12,13,16,17],extern:1,extract:[4,10,16,20],factor:[1,3],factori:1,fals:[1,3,10,13],featur:10,feel:[14,17],few:16,field:[1,9],file:[1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,22],file_ext:17,file_funct:1,filenam:[6,13],filepath:10,filetyp:[1,4,5,7,11,12,13],fill:[3,17],find:[8,13,16,17,18,19,22],fine:16,finish:[9,10,17],first:[1,10,14,16,17,19,20,22],fit:16,fix:[1,10,14],flip:[10,18,20],float64:3,fmri:16,focal:[16,20],folder:[1,2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21],folder_nam:6,follow:[1,2,5,6,9,10,13,15,16,17,18,19,20,22],foreground:3,format:[1,13,14,17,19,21],formatwarn:13,found:16,frac:[13,16,18],fraction:17,fragment:3,framework:2,free:[14,17],frequenc:4,fresh:11,friedmann:22,from:[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,19,20,21,22],full:13,full_plot:16,fulli:[17,20],func:1,fund:22,further:22,futur:16,g:[2,3,4,9,13,14,16,17,22],gener:[16,17,20],generatorwork:4,get:[1,13,16,22],get_available_model:2,get_devic:2,get_loss:10,get_model:2,get_net:15,get_output:15,get_padding_dim:9,get_valid:15,get_weights_fil:15,getter:[2,10],github:[17,20,22],given:[1,4,5,9,11,16,20],go:22,good:16,gpu:22,gradient:16,greatli:[16,22],grid:17,ground:[3,8,13,16,18],group:1,guid:[10,16,22],h:[1,13,15],ha:[1,9,11,13,17,18,20,22],handl:[1,3,16],handler:1,happen:16,hardwar:22,has_result:[2,5],have:[1,10,11,14,16,17,18,19,20,21,22],header:1,height:[1,15],help:22,her:22,here:[4,15,16,22],hide:1,high:[5,16],higher:[16,20],highlight:1,hit:14,horizont:1,hour:13,hour_minute_second:13,hous:22,how:[9,16,22],howev:16,http:22,human:16,i:[11,16],id:[3,16,17,21],ideal:[19,20],imag:[0,1,2,3,4,5,6,7,9,10,11,13,14,17,18,19,20,21,22],image_id:[4,9],image_layer_load:5,image_nam:6,image_path:[5,6,7,11],image_shap:13,images_filepath:[2,4,5],immedi:10,impact:16,implement:[1,5,7,9,11,16,17,20],improv:[16,20],includ:[1,17],increas:[16,20],increment:1,index:[0,4,9,12,19],indic:[4,16,18],inexact:18,infer:[0,2,4,5,18,20,22],inference_result:2,inferencework:9,inferenceworkerconfig:9,infererconfig:9,inferno:17,info:[1,22],inform:17,inherit:[1,4,7,8,9,10,11,13],initi:[1,4,10,12,16,20,22],initializewindow:12,input:[3,9,15,16],instal:[2,16],instanc:[1,3,4,6,9,10,13,16,17,21],instance_config:9,instance_seg:17,instancesegconfig:9,instanti:[1,13],instead:[2,16],instruct:[16,22],insuffici:18,intend:16,intens:[10,20],interest:14,interfac:[0,5],interpret:1,interv:[4,10,16,20],introduct:0,invalid:13,io:[5,22],ipynb:16,is_file_path:3,isol:16,issu:[16,20],item:[1,16],its:[16,17],itself:16,job:4,keep:[1,4,16,17,19,20],keep_on_cpu:4,kei:[2,4,10],know:16,kousi:22,kwarg:[1,3],l:1,label:[0,1,2,3,5,6,7,8,9,10,11,12,13,14,17,18,19,20,22],label_befor:1,label_dir:12,label_layer_load:5,label_path:[5,7,10,11],labels_filepath:[2,5],laboratori:22,lambda_dic:20,larg:[3,13,16,18,20],larger:[16,20],last:[1,16],later:17,launch:[7,9,11,16,17],launch_review:11,layer:[2,4,6,9,16,17,19,21],layerselect:5,layout:[1,5,6,8,16],lead:18,learn:[4,10,16,20],learning_r:4,least:16,leav:[11,16,17],left:[1,2,14,16,17,21],left_or_abov:1,len:13,length:[10,16],less:16,let:[1,11,14,16,20,21],level:5,librari:[16,22],light:22,lightsheet:16,lightweight:16,ligthsheet:16,like:[1,14,16,17,20],line:[1,13],lineno:13,link:[17,20,22],list:[1,2,4,6,11,13,16],littl:16,ll:[16,17,22],ln:13,load:[1,2,3,4,5,6,7,9,10,11,12,13,14,16,17,19,20],load_as_fold:1,load_csv:12,load_dataset_path:5,load_image_dataset:5,load_label_dataset:5,loader:9,loads_imag:[2,5],loads_label:[2,5],local:22,locat:[1,11,13,16,19],lock:1,log:[2,4,10,17,20],log_sign:4,logger:1,longer:[16,20],loop:16,loss:[4,8,10,16,20],loss_dict:10,loss_funct:4,loss_index:10,low:[16,18],lower:[1,9,14,16,17,20],machin:20,mackenzi:22,made:1,mai:[1,14,16,17],main:[1,22],mainli:22,major:[3,13],make:[1,3,14,16,19],make_scrol:1,manag:11,mani:[9,16],manner:1,manual:[19,22],map:[3,16,22],margin:1,mask:3,match:[5,8,10,11,16,18],mathi:22,matter:16,max_epoch:4,max_wh:1,maxim:22,maximum:[1,9,16],mayb:1,mean:[1,4,20],measur:18,medic:[16,17,20],memori:[1,3,10,16,17,20],menu:[1,5,6,7,9,10,11,14],merg:1,mesoscal:22,mesospim:[16,17,20,22],messag:[1,4,10,13],metaclass:[1,13],method:[1,6,16,17,21],metric:[0,4,8,10,20],mice:22,micron:[1,9,17],microscop:[14,16,21,22],microscopi:22,might:[3,16,18,20],min_spac:1,min_wh:1,minimum:[1,17],minor:[3,13],minut:13,mismatch:16,miss:[16,22],mode:[9,17],model:[0,2,4,9,10,12,17,19,20,22],model_dict:[4,15],model_framework:[0,9,10,15,17,20],model_index:10,model_info:9,model_instance_seg:0,model_typ:12,model_work:[0,2,9,10,17,20],modelclass:15,modelinfo:9,models_dict:4,modifi:15,modul:[2,9,10,14,16,22],modular:16,monai:[1,2,4,9,15,16,20,22],monitor:[16,19,22],more:[14,16,17,20],most:[16,17],motor:22,mous:[1,16,22],move:[7,14],mri:[17,20],msg:1,much:20,multipl:[5,16],multithread:[1,10],must:[10,13,15],n:15,name:[1,4,6,9,10,14,15,16,17,18,19],napari:[1,2,4,5,7,9,10,11,12,14,16,17,20,22],napari_cellseg3d:[1,2,3,4,5,6,7,8,9,10,11,12,13,15],nbr_to_show:9,ndarrai:3,nearest:[13,16],need:[4,5,7,9,11,15,16,18,19,22],neg:16,network:[17,20,22],neural:[16,17,20],neuron:16,newest:12,next:[9,10,14,16,20],none:[1,2,5,6,7,9,10,11,12,13],normal:13,note:[4,16,17,20],notebook:17,novel:16,now:[10,16,20],np:3,nuclei:[16,22],num_sampl:4,number:[1,3,4,9,10,13,16,17,18,19,20],numer:[18,20],numpi:[3,13],o:16,object:[0,1,3,6,9,11,13,17,21,22],obtain:[4,10],occupi:16,often:16,on_error:[9,10],on_finish:[9,10],on_lay:9,on_start:[9,10],on_yield:9,onc:[4,10,11,13,14,16,17,18,19,20],one:[1,10,14,16,17,18,19,20],ones:[16,17],onli:[1,5,13,16,17,19,20],opac:14,open:[1,7,11,16,19,22],openurl:1,oper:[2,6,16],opt:14,optim:[4,10,16,20],option:[1,9,14,16,22],order:[1,3],org:22,orient:[8,16,18],origin:[4,6,7,9,11,16,17,20,22],original_nam:17,os:1,other:[1,16,17,22],otherwis:[1,13],our:16,out:[15,16,17],out_path:13,output:[2,5,9,15,16,17],over:[3,16],overhead:1,own:16,p:13,packag:22,pad:[9,13,16,18,20],page:[0,22],pair:[8,16,17,18],panda:12,paper:[17,20],param:[1,6,8],paramet:[1,2,3,4,5,6,9,10,12,13,14,16,17,18,20],parent:[1,2,5,6,7,8,9,11,12],part:16,patch:[4,10,16,20],path:[1,2,3,4,5,6,7,9,10,11,12,13,16,19,20],per:[3,19,21],percentag:[4,16],perfect:16,perform:[4,9,17,20],pi:13,pick:14,pip:22,pixel:[1,3,9,16,17,20,21],place:15,plane:19,pleas:[3,16,17,18,20,22],plot:[8,10,11,16,17,18,19,20],plot_dic:8,plot_loss:10,plugin:[1,2,5,6,7,8,9,10,11,14,16,22],plugin_bas:[0,7,8,11,18,19,21],plugin_convert:[0,21],plugin_crop:0,plugin_dock:[0,19],plugin_metr:[0,18],plugin_model_infer:[0,17],plugin_model_train:[0,20],plugin_review:[0,19],plugin_review_dock:12,pna:22,png:[7,11,13,19],polici:1,port:[16,22],posit:14,possibl:20,possible_path:1,post:9,post_process_config:9,postprocessconfig:9,power:[9,13,16,18,20],pre:[5,10,16,17,20,22],predict:[8,11,13,16,18,19,22],prefer:[16,20],prefix:17,preform:9,prepar:[0,12],present:[7,11,16,20],press:[1,14,16,17,18,19,20],pretrain:[15,17,20],previous:[14,16],print:[1,2],print_and_log:1,probabl:[3,9,16,17],proce:16,process:[3,6,7,9,11,13,16,17,20],produc:17,program:20,progress:[1,2,9,16,17],project:[11,17,19,20,22],prompt:[7,11],proper:[4,15],properli:[9,17,19],properti:1,proport:[10,16,20],provid:[1,3,5,16,17,18,19,20,22],pth:[15,16],put:1,py:[0,15,17,18,19,20,21],pyqt5:12,pytorch:[4,10,15,16,17,20,22],qcheckbox:1,qdesktopservic:1,qfiledialog:1,qinstallmessagehandl:1,qlabel:1,qlayout:[1,8],qlineedit:1,qpushbutton:1,qrect:12,qregion:12,qscrollarea:1,qsizepolici:1,qt:[1,4],qtcore:4,qtextedit:1,qtpy:4,quick:22,quicker:20,quickli:11,quicksav:[7,14,16],qwidget:[1,2,5,9],r:1,rais:13,ram:17,ran:17,random:[10,20],randomli:[10,18],rate:[4,10,16,20],rather:[13,16,17],ratio:[3,16,17],raw:[11,17],re:[5,7,9,11,16,17,20],reach:16,react:1,read:[1,16],reader:22,readi:[14,18,19,20],reason:16,recommend:[20,22],record:[16,19,20],red:[1,18],reduc:16,refer:[8,20],regard:[4,10,22],region:[14,16],regular:[17,20],rel:16,relat:[16,20,22],rem_seed_thr:3,rememb:[16,17],remov:[2,3,5,6,7,9,11,16,17,19,21],remove_docked_widget:5,remove_from_view:[5,7,9,11],remove_plot:8,remove_small_object:3,render:[1,8],repeat:16,replac:1,replace_last_lin:1,report:9,repositori:16,repres:16,reproduc:16,request:[9,17],requir:[1,4,16,17,20],reset:9,resiz:[1,3,14,19,21],resolut:[1,9,14,16,17,21],resolution_xyz:1,resolution_zyx:1,respect:[1,9],restart:16,result:[1,2,3,4,5,6,9,10,13,15,16,17,18,19,20,21],results_path:[2,4,5,6,10],retain:16,review:[0,11,14,22],right:[1,5,16,21],right_or_below:1,rise:16,rng:4,roi:9,rotat:[8,18],run:[4,6,9,16,17,18,21,22],run_method:6,run_review:11,s41592:22,s:[2,4,10,13,14,15,16,22],safe:1,said:13,same:[14,16,17,18,19,20],sampl:[4,10,20,22],sample_s:4,save:[2,4,5,6,7,9,10,11,13,14,16,17,19,20,21,22],save_log:2,save_log_to_path:2,saved_weight:10,scale:[1,3,14,19],scale_factor:3,scaling_xyz:1,scaling_zyx:1,score:[0,18,22],scratch:16,scroll:1,scrollabl:1,search:0,second:[1,10,13,14,20],section:[16,22],see:[2,4,7,9,10,11,14,16,17,19,20,22],seed:[3,4,16,20],segment:[3,4,6,9,10,16,17,20,22],segresnet:[15,16,17,20],select:[1,2,4,5,6,7,9,10,11,14,16,17,20,21],self:[2,6,8,10,11,12],semant:[3,6,16,17,21],semi:[3,13],semi_major:13,semi_minor:13,send:[4,10],send_log:[2,10],sent:4,separ:[4,14,16,17,20],seri:16,session:16,set:[1,5,6,8,9,10,14,16,17,18,20],setfixeds:1,sever:[5,6,7,11,17,19,20,22],shape:[3,13],sheet:22,shift:[10,11,16,17,19,20],shortcut:[1,5],should:[1,4,5,6,9,10,11,13,15,16,17,18,19,20,22],show:[1,2,5,6,13,16,17],show_utils_menu:1,shown:[16,17,18,20],side:[1,2,17,18],sigmoid:20,signal:[2,4,9,10],similar:[16,18],simpli:[14,16,17,20],simultan:[14,16],singl:[1,5,7,12,13,16,19,20],singleton:1,sip:12,situat:16,size:[1,3,4,9,10,11,13,14,15,16,17,18,20,21],skimag:3,slice:[11,12,16,19],slide:9,slider:[6,14,16],sliding_window_infer:15,slow:16,small:[3,6,9,16,18,20,21],smaller:[3,10,14,16,17,18,21],so:[9,16,19,20],softwar:22,somatomotor:17,somatomotor_vnet_2022_04_06_15_49_42_pred3:17,some:[4,16,22],sorensen:13,sort:18,sourc:22,space:1,specif:[2,4,16,17],specifi:[1,6,12,13,16,17,20,21],spheric:[3,13,16,17],spin:[1,10],spinbox:[1,7],sqrt:13,stack:[1,7,13,14,19,20],start:[1,2,9,10,14,16,17,19,20],stat:[4,9,16,17],statist:[3,17],stats_csv:4,statu:[1,9,11,12,16,19,22],step:[1,10,20],stephan:22,stick:16,still:[14,16,19],stop:[10,16,20],store:[12,20],str:[1,2,4,5,7,10,11,12,13],straightforward:16,string:[1,3,13],strongli:22,structur:15,sub:16,subplot:10,subsequ:10,subtract:13,success:22,superior:[13,16],supervis:22,support:[16,17,19,20],sure:[3,14,16],surface_area:13,surround:[16,19],swin:17,swinunetr:17,system:16,t:1,tab:[10,16,20],tabl:[10,20],take:16,taken:[16,19],task:22,templat:[5,6],ten:17,tensor:2,termin:1,test:16,text:[1,2,4,10,16],text_field:1,than:[3,9,10,13,16,17,21],thank:22,thei:[7,16,17,18,19,20],them:[1,3,4,9,14,16,17,18,19,22],themselv:13,therefor:3,thi:[3,11,13,14,16,17,18,19,20,21,22],third:[10,17,20],though:16,thre:3,thread:[1,4],three:[1,7,14,16,19],thres_object:3,thres_seed:3,thres_smal:3,threshold:[3,8,9,14,16,17,18,21],through:10,throughout:[1,14],tif:[1,7,11,13,16,17,19,20],tiff:[1,17,20],time:[1,4,13,14,16,17,19,20],time_finish:13,time_start:13,timokleia:22,tissu:[16,22],titl:1,todo:[1,9,10,11],toggl:[1,10],too:[14,16],tool:[14,16,22],top:1,torch:[4,10,13],total:[3,16,17],track:[11,16,19],trail:19,trailmap:[16,17,20,22],trailmap_m:[16,17,20],train:[0,2,4,5,9,10,17,21,22],train_loss_plot:10,trainer:4,trainingwork:10,transfer:[4,16,20],transform:[2,4,17],treat:3,tri:[9,10],truth:[3,8,13,16,18],tumor:[17,20],tune:16,tupl:3,turbo:17,tutori:16,tverski:20,twilight:17,two:[1,8,9,10,13,16,18,20],type:[1,2,3,4,5,6,7,8,10,11,12,13,15],ui:2,uint:10,under:11,underneath:20,unet:15,uniqu:[3,13,16,21],unnecessari:1,unsur:16,unus:13,up:[1,9,16,17,20],updat:[1,9,10,12,19],update_field_color:1,update_loss_plot:10,upgrad:9,upper:1,url:1,us:[1,2,3,4,5,6,9,10,13,14,16,17,18,19,20,21,22],usabl:16,usag:[5,16,17,20],use_window:4,user:[1,6,10,11],usual:2,util:[1,5,7,9,19,22],v_:13,val_input:15,val_interv:[4,10],val_output:15,valid:[4,10,15,16,17,20],validation_perc:4,validation_step:10,valu:[1,13,16,17,20,21],variabl:16,variou:[3,4,17,21,22],vector:10,veri:[13,16,18,22],verifi:19,version:[7,20,22],versu:[16,20],vertic:1,via:[3,10,22],vidal:22,view:[8,11,16,17],viewer:[2,5,6,7,8,9,10,11,12,14,18],visibility_condit:1,visibl:1,visual:19,vnet:[16,17,20],voidptr:12,voigt:22,volum:[3,4,6,7,9,11,13,14,16,17,20,22],volume_:16,volume_imag:3,volumetr:[17,20],vram:17,vs:17,w:[1,13,15],wa:[6,7,16,17,22],wai:[14,16],walkthrough:0,want:[11,14,16,17,19],warn:[1,10,13],watersh:[3,16,17,21],we:[16,22],websit:22,weight:[4,9,10,15,16,17,20,22],weights_dict:4,weights_fil:15,weights_path:4,weird:1,well:[1,8,14,15,16,17,18,19,20],were:22,what:7,when:[1,2,3,4,10,11,14,16,17,18,20],where:[10,16,17],wherea:[16,17],whether:[1,4,5,6,9,10,11,13,17,18,19,20],which:[2,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,22],whole:[3,4,10,16,17,22],whose:16,widget:[1,2,5,6,7,8,9,10,11,12],width:[1,15],wil:17,window:[1,2,4,5,7,8,9,11,12,16,17],window_infer_s:4,wip:[15,16,22],wish:[16,19],without:[9,13,17,22],work:[5,11,16,21],workabl:20,worker:[2,4,9,10,17],worker_config:[4,9],workerbasesign:4,workflow:16,workspac:9,write:[1,11],written:[17,22],wrong:18,wyss:22,x1:15,x2:15,x:[1,3,7,11,14,16,17,18,19],xyz:1,y:[1,3,7,11,14,16,17,18,19],y_pred:13,y_true:13,year_month_day_hour_minute_second:13,yet:20,yield:[4,9,10,16,17],you:[11,14,15,16,17,18,19,20,21,22],your:[14,15,16,17,18,19,20,21,22],z:[1,3,7,11,14,16,17,19],zero:13,zip:[16,20],zoom:14,zyx:1},titles:["Welcome to napari-cellseg3d\u2019s documentation!","interface.py","model_framework.py","model_instance_seg.py","model_workers.py","plugin_base.py","plugin_convert.py","plugin_crop.py","plugin_metrics.py","plugin_model_inference.py","plugin_model_training.py","plugin_review.py","plugin_dock.py","utils.py","Cropping utility guide","Advanced : Declaring a custom model","Detailed walkthrough","Inference module guide","Metrics utility guide","Review module guide","Training module guide","Label conversion utility guide","Introduction"],titleterms:{"class":[1,2,4,5,6,7,8,9,10,11,13],"function":[1,3,6,13,14,17,19],"new":14,acknowledg:22,add_blank:1,add_label:1,advanc:[0,15],analysi:16,anisotropywidget:1,anisoutil:6,attribut:[2,4,5,7,8,9,10,11,12],basepluginfold:5,basepluginsingleimag:5,binary_connect:3,binary_watersh:3,button:1,cellseg3d:0,checkbox:1,clear_small_object:3,code:[17,18,19,20,21],combine_block:1,containerwidget:1,convers:[16,21],convert:16,creat:14,crop:[7,14,16],custom:15,datamanag:12,declar:15,denormalize_i:13,detail:16,detect:16,dice_coeff:13,document:0,doubleincrementcount:1,dropdownmenu:1,file:0,filepathwidget:1,format_warn:13,get_date_tim:13,get_padding_dim:13,get_tim:13,get_time_filepath:13,guid:[0,14,17,18,19,20,21],handle_adjust_error:1,handle_adjust_errors_wrapp:1,imag:16,indic:0,infer:[9,16,17],inferencework:4,instal:22,instancewidget:6,interfac:[1,14,17,19],intincrementcount:1,introduct:22,jupyt:16,label:[16,21],launch:[14,19],layer:14,load_imag:13,loader:11,log:1,logsign:4,main:0,make_group:1,method:[2,4,5,7,8,9,10,11,12],metric:[16,18],metricsutil:8,model:[15,16],model_framework:2,model_instance_seg:3,model_work:4,modelframework:2,modul:[0,17,19,20],napari:0,normalize_i:13,normalize_x:13,notebook:16,object:16,open_file_dialog:1,open_url:1,perform:16,plugin_bas:5,plugin_convert:6,plugin_crop:7,plugin_dock:12,plugin_metr:8,plugin_model_infer:9,plugin_model_train:10,plugin_review:11,prepar:16,process:[14,19],py:[1,2,3,4,5,6,7,8,9,10,11,12,13],qwidgetsingleton:1,refer:22,removesmallutil:6,requir:22,review:[16,19],s:0,save_fold:6,save_lay:6,save_stack:13,score:16,scrollarea:1,show_result:6,singleton:13,sourc:[0,17,18,19,20,21],sphericity_axi:13,sphericity_volume_area:13,tabl:0,thresholdutil:6,through:0,time_differ:13,to_inst:3,to_semant:3,toggle_vis:1,toinstanceutil:6,tosemanticutil:6,train:[16,20],trainer:10,trainingwork:4,usag:22,util:[0,13,14,16,18,21],utilsdropdown:1,volume_stat:3,walk:0,walkthrough:16,welcom:0}})