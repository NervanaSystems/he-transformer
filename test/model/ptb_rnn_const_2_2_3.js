[{"name":"Function_1","ops":[{"element_type":"float","inputs":[],"name":"Parameter_312","op":"Parameter","outputs":["Parameter_312_0"],"shape":[50]},{"element_type":"float","inputs":[],"name":"Parameter_287","op":"Parameter","outputs":["Parameter_287_0"],"shape":[3,50]},{"element_type":"float","inputs":[],"name":"Parameter_285","op":"Parameter","outputs":["Parameter_285_0"],"shape":[3]},{"element_type":"float","inputs":[],"name":"Parameter_281","op":"Parameter","outputs":["Parameter_281_0"],"shape":[3,3]},{"element_type":"float","inputs":[],"name":"Parameter_280","op":"Parameter","outputs":["Parameter_280_0"],"shape":[50,3]},{"element_type":"float","inputs":[],"name":"Parameter_277","op":"Parameter","outputs":["Parameter_277_0"],"shape":[2,2]},{"element_type":"float","inputs":[],"name":"Parameter_288","op":"Parameter","outputs":["Parameter_288_0"],"shape":[2,2]},{"element_type":"float","inputs":[],"name":"Constant_328","op":"Constant","outputs":["Constant_328_0"],"shape":[],"value":["1.4427"]},{"element_type":"float","inputs":[],"name":"Constant_325","op":"Constant","outputs":["Constant_325_0"],"shape":[],"value":["50"]},{"element_type":"float","inputs":[],"name":"Constant_282","op":"Constant","outputs":["Constant_282_0"],"shape":[],"value":["0"]},{"axes":[1,2],"inputs":["Parameter_312"],"name":"Broadcast_313","op":"Broadcast","outputs":["Broadcast_313_0"],"shape":[50,2,2]},{"input_order":[0,1],"inputs":["Parameter_287"],"name":"Reshape_290","op":"Reshape","output_shape":[3,50],"outputs":["Reshape_290_0"]},{"axes":[1],"inputs":["Parameter_285"],"name":"Broadcast_286","op":"Broadcast","outputs":["Broadcast_286_0"],"shape":[3,2]},{"input_order":[0,1],"inputs":["Parameter_280"],"name":"Reshape_308","op":"Reshape","output_shape":[50,3],"outputs":["Reshape_308_0"]},{"inputs":["Parameter_277"],"name":"OneHot_278","one_hot_axis":0,"op":"OneHot","outputs":["OneHot_278_0"],"shape":[50,2,2]},{"inputs":["Parameter_288"],"name":"OneHot_289","one_hot_axis":0,"op":"OneHot","outputs":["OneHot_289_0"],"shape":[50,2,2]},{"axes":[0,1],"inputs":["Constant_328"],"name":"Broadcast_329","op":"Broadcast","outputs":["Broadcast_329_0"],"shape":[2,2]},{"axes":[0,1],"inputs":["Constant_325"],"name":"Broadcast_326","op":"Broadcast","outputs":["Broadcast_326_0"],"shape":[2,2]},{"axes":[0,1],"inputs":["Constant_282"],"name":"Broadcast_283","op":"Broadcast","outputs":["Broadcast_283_0"],"shape":[3,2]},{"axes":[],"inputs":["OneHot_278"],"name":"Broadcast_279","op":"Broadcast","outputs":["Broadcast_279_0"],"shape":[50,2,2]},{"input_order":[0,1,2],"inputs":["OneHot_289"],"name":"Reshape_291","op":"Reshape","output_shape":[50,4],"outputs":["Reshape_291_0"]},{"inputs":["Parameter_281","Broadcast_283"],"name":"Dot_284","op":"Dot","outputs":["Dot_284_0"],"reduction_axes_count":1},{"inputs":["Reshape_290","Reshape_291"],"name":"Dot_292","op":"Dot","outputs":["Dot_292_0"],"reduction_axes_count":1},{"input_order":[0,1],"inputs":["Dot_292"],"name":"Reshape_293","op":"Reshape","output_shape":[3,2,2],"outputs":["Reshape_293_0"]},{"inputs":["Reshape_293"],"lower_bounds":[0,0,0],"name":"Slice_294","op":"Slice","outputs":["Slice_294_0"],"strides":[1,1,1],"upper_bounds":[3,2,1]},{"inputs":["Reshape_293"],"lower_bounds":[0,0,1],"name":"Slice_301","op":"Slice","outputs":["Slice_301_0"],"strides":[1,1,1],"upper_bounds":[3,2,2]},{"input_order":[0,1,2],"inputs":["Slice_294"],"name":"Reshape_295","op":"Reshape","output_shape":[3,2],"outputs":["Reshape_295_0"]},{"input_order":[0,1,2],"inputs":["Slice_301"],"name":"Reshape_302","op":"Reshape","output_shape":[3,2],"outputs":["Reshape_302_0"]},{"inputs":["Dot_284","Reshape_295"],"name":"Add_296","op":"Add","outputs":["Add_296_0"]},{"inputs":["Add_296","Broadcast_286"],"name":"Add_297","op":"Add","outputs":["Add_297_0"]},{"inputs":["Add_297","Add_297"],"name":"Multiply_298","op":"Multiply","outputs":["Multiply_298_0"]},{"axes":[1],"inputs":["Multiply_298"],"name":"Broadcast_299","op":"Broadcast","outputs":["Broadcast_299_0"],"shape":[3,1,2]},{"inputs":["Parameter_281","Multiply_298"],"name":"Dot_300","op":"Dot","outputs":["Dot_300_0"],"reduction_axes_count":1},{"inputs":["Dot_300","Reshape_302"],"name":"Add_303","op":"Add","outputs":["Add_303_0"]},{"inputs":["Add_303","Broadcast_286"],"name":"Add_304","op":"Add","outputs":["Add_304_0"]},{"inputs":["Add_304","Add_304"],"name":"Multiply_305","op":"Multiply","outputs":["Multiply_305_0"]},{"axes":[1],"inputs":["Multiply_305"],"name":"Broadcast_306","op":"Broadcast","outputs":["Broadcast_306_0"],"shape":[3,1,2]},{"axis":1,"inputs":["Broadcast_299","Broadcast_306"],"name":"Concat_307","op":"Concat","outputs":["Concat_307_0"]},{"input_order":[0,1,2],"inputs":["Concat_307"],"name":"Reshape_309","op":"Reshape","output_shape":[3,4],"outputs":["Reshape_309_0"]},{"inputs":["Reshape_308","Reshape_309"],"name":"Dot_310","op":"Dot","outputs":["Dot_310_0"],"reduction_axes_count":1},{"input_order":[0,1],"inputs":["Dot_310"],"name":"Reshape_311","op":"Reshape","output_shape":[50,2,2],"outputs":["Reshape_311_0"]},{"inputs":["Reshape_311","Broadcast_313"],"name":"Add_314","op":"Add","outputs":["Add_314_0"]},{"inputs":["Add_314"],"name":"Max_315","op":"Max","outputs":["Max_315_0"],"reduction_axes":[0]},{"axes":[0],"inputs":["Max_315"],"name":"Broadcast_316","op":"Broadcast","outputs":["Broadcast_316_0"],"shape":[50,2,2]},{"inputs":["Add_314","Broadcast_316"],"name":"Subtract_317","op":"Subtract","outputs":["Subtract_317_0"]},{"inputs":["Subtract_317","Broadcast_279"],"name":"Multiply_318","op":"Multiply","outputs":["Multiply_318_0"]},{"inputs":["Subtract_317"],"name":"Exp_321","op":"Exp","outputs":["Exp_321_0"]},{"inputs":["Multiply_318"],"name":"Sum_319","op":"Sum","outputs":["Sum_319_0"],"reduction_axes":[0]},{"inputs":["Exp_321"],"name":"Sum_322","op":"Sum","outputs":["Sum_322_0"],"reduction_axes":[0]},{"inputs":["Sum_319"],"name":"Negative_320","op":"Negative","outputs":["Negative_320_0"]},{"inputs":["Sum_322"],"name":"Log_323","op":"Log","outputs":["Log_323_0"]},{"inputs":["Negative_320","Log_323"],"name":"Add_324","op":"Add","outputs":["Add_324_0"]},{"inputs":["Add_324","Broadcast_326"],"name":"Minimum_327","op":"Minimum","outputs":["Minimum_327_0"]},{"inputs":["Minimum_327","Broadcast_329"],"name":"Multiply_330","op":"Multiply","outputs":["Multiply_330_0"]},{"inputs":["Multiply_330"],"name":"Result_331","op":"Result","outputs":["Result_331_0"]}],"parameters":["Parameter_288","Parameter_277","Parameter_280","Parameter_281","Parameter_285","Parameter_287","Parameter_312"],"result":["Result_331"]}]
