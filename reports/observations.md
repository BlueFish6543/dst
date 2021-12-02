# Baseline experiment

Investigating unseen services:

* Wrong slot names
  * E.g. Predicting `city` instead of `location` for `Restaurants_2` (learned from `Restaurants_1`)
  * E.g. Predicting `party_size` instead of `number_of_seats` for `Restaurants_2` (learned from `Restaurants_1`)
  * E.g. Predicting `song_name` instead of `track` for `Music_3` (learned from `Music_1` and `Music_2`)
  * E.g. Predicting `city_of_event` instead of `city` for `Events_3` (learned from `Events_1`)
  * E.g. Predicting `number_of_seats` instead of `number_of_tickets` for `Events_3` (learned from `Events_1`)
  * E.g. Predicting `category` instead of `event_type` for `Events_3` (learned from `Events_1`)
* Issues with categorical slots
  * E.g. `playback_device` slot in `Music_1` and `Music_2` has a set of categorical values. Model learns to predict
    one of those. But in unseen `Music_3` the slot is renamed to `device` and has a different set of categorical values.
* Fail to predict a certain slot
  * Might be linked to categorical values. For instance `event_type` has possible values `Music, Sports` in
    `Events_2` but has possible values `Music, Theater` in unseen `Events_3`. Model fails to predict `Theater`
  * Non-existent slots in equivalent seen services don't seem to be predicted.
    * They may cause model to get confused. E.g. `intent` slot in unseen
    `Homes_2` somehow causes model to predict `property_name` slot
* For unseen domain `Payments_1` the model substitutes with seen `Banks_1`. Because we match the predicted service
  with the most similar service (in terms of slots), this itself is not a problem
  * However, the correspondence between slots is not one-to-one. So naturally the model does not know how to predict
    new slots in `Payments_1`
* For unseen domain `Messaging_1` the model completely misses it as there is no equivalent seen domain/service
* `Services_X` services are all quite different with different slots so the model has difficulty generalising to
  `Service_4` (hair stylist, dentist, doctor, therapist)
  * E.g. Predicting `doctor_name` slot instead of `therapist_name`
* Multi-service dialogues
  * Predicting the wrong service
  * Only predicting one out of multiple services
* Predicting wrong service (e.g. `Hotels` instead of `Homes`), which causes slot names to be wrong

Investigating seen serivces:

* Conflicting slot names between services in same domain. E.g. `number_of_riders` and `number_of_seats` in
  `RideSharing_1` and `RideSharing_2`